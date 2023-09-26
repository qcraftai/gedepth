from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import pre_eval_to_metrics, metrics, eval_metrics
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from depth.ops import resize

from PIL import Image
import cv2
import torch
from IPython import embed
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import sys

@DATASETS.register_module()
class DDADDataset(Dataset):
    def __init__(self,
                 pipeline,
                 cameras = ['CAMERA_01','CAMERA_05','CAMERA_06','CAMERA_07','CAMERA_08','CAMERA_09'],
                 split=None,
                 test_mode=False,
                 garg_crop=False,
                 eigen_crop=False,
                 min_depth=1e-3,
                 max_depth=200):
        self.garg_crop = garg_crop
        self.eigen_crop = eigen_crop
        self.cameras = cameras
        self.pipeline = Compose(pipeline)
        self.split = split
        self.test_mode = test_mode
        self.min_depth = min_depth # just for evaluate. (crop gt to certain range)
        self.max_depth = max_depth # just for evaluate.

        # load annotations
        self.img_infos = self.load_annotations(self.split)

        self.cam_k = {
            'CAMERA_01' : np.array([[2.1815303e+03, 0.0000000e+00, 9.2802191e+02, 0], 
                            [0.0000000e+00, 2.1816035e+03, 6.1595679e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0],
                            [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            'CAMERA_05' : np.array([[1.0570685e+03, 0.0000000e+00, 9.6468347e+02, 0], 
                            [0.0000000e+00, 1.0559746e+03, 5.8866125e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0],
                            [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            'CAMERA_06' : np.array([[1.0607557e+03, 0.0000000e+00, 9.4655847e+02, 0], 
                            [0.0000000e+00, 1.0592549e+03, 6.1140710e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0],
                            [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            'CAMERA_09' : np.array([[1.0634580e+03, 0.0000000e+00, 9.4466577e+02, 0], 
                            [0.0000000e+00, 1.0652224e+03, 6.1269843e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0],
                            [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
        }
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            split (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    depth_map = line.strip().split(" ")[1]
                    if depth_map.split('/')[-2] in self.cameras:
                        img_info['ann'] = dict(depth_map=depth_map.replace('depth_val','depth'))
                        img_name = line.strip().split(" ")[0]
                        img_info['filename'] = img_name
                        img_infos.append(img_info)
        else:
            print("Split should be specified, NotImplementedError")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())

        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['cam_intrinsic_dict'] = {
            'CAMERA_01' : [[2.1815303e+03, 0.0000000e+00, 9.2802191e+02, 0], 
                            [0.0000000e+00, 2.1816035e+03, 6.1595679e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0]],
            'CAMERA_05' : [[1.0570685e+03, 0.0000000e+00, 9.6468347e+02, 0], 
                            [0.0000000e+00, 1.0559746e+03, 5.8866125e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0]],
            'CAMERA_06' : [[1.0607557e+03, 0.0000000e+00, 9.4655847e+02, 0], 
                            [0.0000000e+00, 1.0592549e+03, 6.1140710e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0]],
            'CAMERA_09' : [[1.0634580e+03, 0.0000000e+00, 9.4466577e+02, 0], 
                            [0.0000000e+00, 1.0652224e+03, 6.1269843e+02, 0],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0]],
        }

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        ##
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        ##
        # results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        results[0] = (results[0]).astype(np.uint16)
        return results

    def get_gt_depth_maps(self):
        """Get ground truth depth maps for evaluation."""
        for img_info in self.img_infos:
            depth_map_gt = np.load(img_info['ann']['depth_map'])['depth']
            yield depth_map_gt
    
    def eval_kb_crop(self, depth_gt):
        """Following Adabins, Do kb crop for testing"""
        height = depth_gt.shape[0]
        width = depth_gt.shape[1]
        top_margin = int(height - 1200)
        left_margin = int((width - 1936) / 2)
        depth_cropped = depth_gt[top_margin: top_margin + 1200, left_margin: left_margin + 1936]
        depth_cropped = np.expand_dims(depth_cropped, axis=0)
        return depth_cropped
    
    def eval_resize(self, depth):
        """Following Adabins, Do kb crop for testing"""
        depth_cropped = np.expand_dims(depth, axis=0)
        return depth_cropped

    def eval_mask(self, depth_gt):
        """Following Adabins, Do grag_crop or eigen_crop for testing"""
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask
    
    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the depth estimation.
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []
        for i, (pred, index) in enumerate(zip(preds, indices)):
            depth_map = self.img_infos[index]['ann']['depth_map']
            depth_map_gt = np.load(depth_map)['depth'].astype(np.float32)
            depth_map_gt = self.eval_resize(depth_map_gt)
            pred = torch.from_numpy(pred).unsqueeze(0)
            depth_map_gt = torch.from_numpy(depth_map_gt).squeeze(0)
            pred = F.interpolate(pred, 
                                size=depth_map_gt.shape, 
                                mode='bilinear',
                                align_corners=True).squeeze(0).numpy()
            depth_map_gt = depth_map_gt.unsqueeze(0).numpy()
            valid_mask = self.eval_mask(depth_map_gt)
            eval = metrics(depth_map_gt[valid_mask], 
                           pred[valid_mask], 
                           min_depth=self.min_depth,
                           max_depth=self.max_depth)
            pre_eval_preds.append(pred)
            # save prediction results
            
            pre_eval_results.append(eval)
        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict depth map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            gt_depth_maps = self.get_gt_depth_maps()
            ret_metrics = eval_metrics(
                gt_depth_maps,
                results)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results)

        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)


        eval_num = 9

        num_table = len(ret_metrics) // eval_num
        for i in range(num_table):
            names = ret_metric_names[i*eval_num: i*eval_num + eval_num]
            values = ret_metric_values[i*eval_num: i*eval_num + eval_num]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value
        return eval_results