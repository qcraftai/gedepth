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

from depth.core import pre_eval_to_metrics, metrics, eval_metrics,BackprojectDepth
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from depth.ops import resize

from PIL import Image
import cv2
import torch
from IPython import embed
from collections import defaultdict
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

class RMSEEvaluator:
    def __init__(self):
        self.rmse = []
    def update(self,pred,gt):
        self.rmse.append(pow(gt-pred,2))
    def evaluate(self):
        rmse = sum(self.rmse)
        return np.sqrt(rmse)

class AccuracyEvaluator:
    def __init__(self):
        self.tp = 0
        self.total = 0
    def update(self,pred,gt):
        if pred == gt:
            self.tp+=1
        self.total +=1
    def evaluate(self):
        return self.tp/self.total




class MIoUEvaluator:
    """
    Evaluate semantic segmentation accuracy in metric "miou"
    """
    def __init__(self,semantic_class,ignore_label=255,**kwargs):
        """
        Args:
            semantic_class (List): names of semantic classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
        """
        
        self.semantic_class = semantic_class
        self._ignore_label = ignore_label
        self._num_classes = len(semantic_class)
        self._N = self._num_classes + 1

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)

    def update(self,pred,gt):
        pred = pred.astype(np.int64)
        gt = gt.astype(np.int64)
        gt[gt == self._ignore_label] = self._num_classes
        self._conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
        ).reshape(self._N, self._N)
    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
            Mean intersection-over-union averaged across classes (mIoU)
        """
        acc = np.zeros(self._num_classes, dtype=np.float64)
        iou = np.zeros(self._num_classes, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        
        miou = (np.sum(iou) / np.sum(iou_valid))*100
        res = {}
        # res[self.semantic_class[1]] = iou[1]*100
        res["mIoU"] = miou
        for i in range(self._num_classes):
            res[self.semantic_class[i]] = iou[i]*100
        return res


@DATASETS.register_module()
class KITTIDataset(Dataset):
    """KITTI dataset for depth estimation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── KITTI
        │   │   ├── kitti_eigen_train.txt
        │   │   ├── kitti_eigen_test.txt
        │   │   ├── input (RGB, img_dir)
        │   │   │   ├── date_1
        │   │   │   ├── date_2
        │   │   │   |   ...
        │   │   │   |   ...
        |   │   ├── gt_depth (ann_dir)
        │   │   │   ├── date_drive_number_sync
    split file format:
    input_image: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png 
    gt_depth:    2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png 
    focal:       721.5377 (following the focal setting in BTS, but actually we do not use it)
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        ann_dir (str, optional): Path to annotation directory. Default: None
        split (str, optional): Split txt file. Split should be specified, only file in the splits will be loaded.
        data_root (str, optional): Data root for img_dir/ann_dir. Default: None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        depth_scale=256: Default KITTI pre-process. divide 256 to get gt measured in meters (m)
        garg_crop=True: Following Adabins, use grag crop to eval results.
        eigen_crop=False: Another cropping setting.
        min_depth=1e-3: Default min depth value.
        max_depth=80: Default max depth value.
    """


    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 depth_scale=256,
                 garg_crop=True,
                 eigen_crop=False,
                 min_depth=1e-3,
                 max_depth=80,
                 mask_pe = False,
                 mask_pe_gt = False,
                 eval_chamfer=False):

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.garg_crop = garg_crop
        self.eigen_crop = eigen_crop
        self.min_depth = min_depth # just for evaluate. (crop gt to certain range)
        self.max_depth = max_depth # just for evaluate.
        self.mask_pe = mask_pe
        self.mask_pe_gt = mask_pe_gt
        self.iou_over_60_list = []
        self.iou_over_75_list = []
        self.RMSE = RMSEEvaluator()
        self.acc = AccuracyEvaluator()
        self.eval_chamfer = eval_chamfer

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_dir is None or osp.isabs(self.img_dir)):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.ann_dir, self.split)
        self.MIoUEvaluator = MIoUEvaluator(semantic_class=['Mask Value 0 IoU','Mask Value 1 IoU'])
        self.cam_k = {
            '2011_09_26' : np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], 
                                     [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                                     [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),

            '2011_09_28' : np.array([[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01], 
                            [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03],
                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            '2011_09_29' : np.array([[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], 
                            [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03],
                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            '2011_09_30' : np.array([[7.070912e+02, 0.000000e+00, 6.018873e+02, 4.688783e+01], 
                            [0.000000e+00, 7.070912e+02, 1.831104e+02, 1.178601e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 6.203223e-03],
                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            '2011_10_03' : np.array([[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01], 
                            [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03],
                                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]),
        }
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, ann_dir, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            split (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """

        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    if ann_dir is not None: # benchmark test or unsupervised future
                        depth_map = line.strip().split(" ")[1]
                        if depth_map == 'None':
                            self.invalid_depth_num += 1
                            continue
                        img_info['ann'] = dict(depth_map=depth_map)
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = img_name
                    img_infos.append(img_info)
        else:
            print("Split should be specified, NotImplementedError")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered', logger=get_root_logger())

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
        results['img_prefix'] = self.img_dir
        results['depth_prefix'] = self.ann_dir
        results['depth_scale'] = self.depth_scale
        results['cam_intrinsic_dict_for_nromal'] = {
            '2011_09_26' : np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02], 
                            [0.000000e+00, 7.215377e+02, 1.728540e+02],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            '2011_09_28' : np.array([[7.070493e+02, 0.000000e+00, 6.040814e+02], 
                            [0.000000e+00, 7.070493e+02, 1.805066e+02], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            '2011_09_29' : np.array([[7.183351e+02, 0.000000e+00, 6.003891e+02], 
                            [0.000000e+00, 7.183351e+02, 1.815122e+02],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            '2011_09_30' : np.array([[7.070912e+02, 0.000000e+00, 6.018873e+02], 
                            [0.000000e+00, 7.070912e+02, 1.831104e+02], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]]),
            '2011_10_03' : np.array([[7.188560e+02, 0.000000e+00, 6.071928e+02], 
                            [0.000000e+00, 7.188560e+02, 1.852157e+02], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00]]),
        }
        results['cam_intrinsic_dict'] = {
            '2011_09_26' : [[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], 
                            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]],
            '2011_09_28' : [[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01], 
                            [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]],
            '2011_09_29' : [[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], 
                            [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]],
            '2011_09_30' : [[7.070912e+02, 0.000000e+00, 6.018873e+02, 4.688783e+01], 
                            [0.000000e+00, 7.070912e+02, 1.831104e+02, 1.178601e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 6.203223e-03]],
            '2011_10_03' : [[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01], 
                            [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]],
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
        results[0] = (results[0] * self.depth_scale).astype(np.uint16)
        return results

    def get_gt_depth_maps(self):
        """Get ground truth depth maps for evaluation."""
        for img_info in self.img_infos:
            depth_map = osp.join(self.ann_dir, img_info['ann']['depth_map'])
            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            yield depth_map_gt
    
    def eval_kb_crop(self, depth_gt):
        """Following Adabins, Do kb crop for testing"""
        if not self.mask_pe:
            height = depth_gt.shape[0]
            width = depth_gt.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth_cropped = depth_gt[top_margin: top_margin + 352, left_margin: left_margin + 1216]
            depth_cropped = np.expand_dims(depth_cropped, axis=0)
        else:
            depth_cropped = np.expand_dims(depth_gt, axis=0)
        return depth_cropped

    def eval_mask(self, depth_gt):
        """Following Adabins, Do grag_crop or eigen_crop for testing"""
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        if self.garg_crop or self.eigen_crop:
            gt_height, gt_width = depth_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if self.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif self.eigen_crop:
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
        valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask

    def maskpe_eval(self,depth_map,depth_map_gt,pred):
        pe_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input"
        pe_depth = cv2.imread(pe_root_path+'/'+depth_map.split('/')[3].split('_')[0]+'_'+depth_map.split('/')[3].split('_')[1]+'_'+depth_map.split('/')[3].split('_')[2]+'/'+depth_map.split('/')[3]+'/pe.png',-1)/100
        pe_depth[pe_depth>85] = 0
        pe_depth = self.eval_kb_crop(pe_depth)
        gt_err = np.divide(abs(pe_depth-depth_map_gt),depth_map_gt,out=np.zeros_like(depth_map_gt),where=depth_map_gt!=0)
        gt_mask_err = gt_err<=0.03
        gt_mask = np.logical_and(gt_mask_err,depth_map_gt)
            
        gt_mask = gt_mask.astype(np.uint8)
        temp_gt_mask = gt_mask.copy()
        kernel = np.ones((7, 7), 'uint8')
        gt_mask[0] = cv2.dilate(gt_mask[0], kernel, iterations=1)
        #pe_mask = gt_mask.astype(np.uint8)
        pe_mask = np.zeros_like(gt_mask).astype(np.uint8)
        contours,_ = cv2.findContours(gt_mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) > 1:
                cv2.fillConvexPoly(pe_mask[0], cv2.convexHull(contour,clockwise=True,returnPoints=True), (1))
        # kernel = np.ones((9, 9), 'uint8')
        # pe_mask[0] = cv2.erode(pe_mask[0],kernel,iterations = 1)

        # un = abs(pe_mask-temp_gt_mask)

        # pe_mask[un==1] = 2
        # vis_pred = np.zeros((pred[0].shape[0],pred[0].shape[1],3)).astype(np.uint8)
        # vis_pe_mask = np.zeros((pe_mask[0].shape[0],pe_mask[0].shape[1],3)).astype(np.uint8)
        # # vis_pred[pred[0]==2] = [0,0,255]
        # vis_pred[pred[0]==1] = [255,255,255]
        # # vis_pe_mask[pe_mask[0]==2] = [0,0,255]
        # vis_pe_mask[depth_map_gt[0]>0] = [255,255,255]
        # cv2.imwrite('/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/gt_mask_out.png',vis_pe_mask)
        # cv2.imwrite('/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/pred_out.png',vis_pred)
        self.MIoUEvaluator.update(pred.astype(np.uint8),pe_mask)
        pred_out = pred.copy()
        pred = pred*pe_depth
        valid_mask = np.logical_and(depth_map_gt,pred) * depth_map_gt
        valid_mask = self.eval_mask(valid_mask)
        return valid_mask,pred,pred_out,pe_mask
    
    def maskgt_eval(self,depth_map,depth_map_gt,pred):
        pe_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input"
        pe_depth = cv2.imread(pe_root_path+'/'+depth_map.split('/')[3].split('_')[0]+'_'+depth_map.split('/')[3].split('_')[1]+'_'+depth_map.split('/')[3].split('_')[2]+'/'+depth_map.split('/')[3]+'/pe.png',-1)/100
        pe_depth[pe_depth>85] = 0
        pe_depth[pe_depth<0.001] = 0
        pe_depth = self.eval_kb_crop(pe_depth)
        gt_err = np.divide(abs(pe_depth-depth_map_gt),depth_map_gt,out=np.zeros_like(depth_map_gt),where=depth_map_gt!=0)
        gt_mask_err = gt_err<=0.03
        gt_mask = np.logical_and(gt_mask_err,depth_map_gt)
        gt_mask = gt_mask.astype(np.uint8)
        kernel = np.ones((7, 7), 'uint8')
        gt_mask[0] = cv2.dilate(gt_mask[0], kernel, iterations=1)
        pe_mask = np.zeros_like(gt_mask).astype(np.uint8)
        contours,_ = cv2.findContours(gt_mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) > 1:
                cv2.fillConvexPoly(pe_mask[0], cv2.convexHull(contour,clockwise=True,returnPoints=True), (1))
        kernel = np.ones((9, 9), 'uint8')
        pe_mask[0] = cv2.erode(pe_mask[0],kernel,iterations = 1)
        pred = np.logical_and(pred,pe_mask)
        pred = pe_mask-pred
        self.MIoUEvaluator.update(pred.astype(np.uint8),pe_mask)
        pred = pred*pe_depth
        valid_mask = np.logical_and(depth_map_gt,pred) * depth_map_gt
        valid_mask = self.eval_mask(valid_mask)
        return valid_mask,pred
    def eval_each_mask_iou(self,pred,pe_gt):
        EachMIoUEvaluator = MIoUEvaluator(semantic_class=['Mask Value 0 IoU','Mask Value 1 IoU'])
        EachMIoUEvaluator.update(pred,pe_gt[0])
        result = EachMIoUEvaluator.evaluate()
        return result['Mask Value 1 IoU']
        # if 60<=result['Mask Value 1 IoU']:
        #     self.iou_over_60_list.append(result['Mask Value 1 IoU'])
        # if 75<=result['Mask Value 1 IoU']:
        #     self.iou_over_75_list.append(result['Mask Value 1 IoU'])

    def kb_intr(self,K,kh,kw):
        height = kh
        width = kw
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        K[0][2] = K[0][2] - top_margin
        K[1][2] = K[1][2] - left_margin
        return K
        # depth_cropped = depth_gt[top_margin: top_margin + 352, left_margin: left_margin + 1216]


    def points_set_distance(self, src, dst, criterion_mode="l2"):
        """
        Compute the distance from each src point to dst set, and verse
        Args:
            src (torch.Tensor): Source set with shape [B, N, C]
            dst (torch.Tensor): Destination set with shape [B, M, C]
            criterion_mode (str): Criterion mode to calculate distance.
                The valid modes are smooth_l1, l1 or l2
        """
        if criterion_mode == "smooth_l1":
            criterion = smooth_l1_loss
        elif criterion_mode == "l1":
            criterion = l1_loss
        elif criterion_mode == "l2":
            criterion = mse_loss
        else:
            raise NotImplementedError
        src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
        dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)

        distance = criterion(src_expand, dst_expand, reduction="none").sum(-1)
        src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
        dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

        return src2dst_distance, dst2src_distance, indices1, indices2



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
        # iou_over_60_list = []
        # iou_over_75_list = []
        for i, (pred, index) in enumerate(zip(preds, indices)):
            
            depth_map = osp.join(self.ann_dir,
                               self.img_infos[index]['ann']['depth_map'])

            depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
            kh,kw = depth_map_gt.shape
            depth_map_gt = self.eval_kb_crop(depth_map_gt)
            valid_mask = self.eval_mask(depth_map_gt)

            if self.mask_pe:
                valid_mask,pred,pred_out,pe_gt = self.maskpe_eval(depth_map,depth_map_gt,pred)
                mask_iou = self.eval_each_mask_iou(pred_out,pe_gt)
                if mask_iou>=75:
                    self.iou_over_60_list.append(mask_iou)
                    self.iou_over_75_list.append(mask_iou)
                if 60<=mask_iou<75:
                    self.iou_over_60_list.append(mask_iou)
            elif self.mask_pe_gt:
                valid_mask,pred = self.maskgt_eval(depth_map,depth_map_gt,pred)
            eval = metrics(depth_map_gt[valid_mask], 
                           pred[valid_mask], 
                           min_depth=self.min_depth,
                           max_depth=self.max_depth)
            

            # save prediction results
            pre_eval_preds.append(pred)
            if self.eval_chamfer:
                # chamfer distance, precision, recall, F-score
                # 1. generate 3d points
                B,h,w = pred.shape
                back_project_depth = BackprojectDepth(B, h, w)
                back_project_depth.eval()
                K = self.cam_k[self.img_infos[index]['filename'].split('/')[-5]]
                K = self.kb_intr(K,kh,kw)
                K = torch.from_numpy(K.astype(np.float32)).to(torch.cuda.current_device()).unsqueeze(0)
                pred = torch.from_numpy(pred).to(torch.cuda.current_device())
                depth_map_gt = torch.from_numpy(depth_map_gt).to(torch.cuda.current_device())
                inv_K = torch.linalg.inv(K)
                gt_points = back_project_depth(depth=depth_map_gt, inv_K=inv_K)[:, :3, :]
                gt_points = gt_points.permute(0, 2, 1)
                pred_points = back_project_depth(depth=pred, inv_K=inv_K)[:, :3, :]
                pred_points = pred_points.permute(0, 2, 1)
                mask_3d = (gt_points[:, :, 2] > 0).squeeze(0)
                gt_points = gt_points[:, mask_3d, :]
                pred_points = pred_points[:, mask_3d, :]

                # 2.compute distance from a single point to points set
                batch_size_ = int(gt_points.shape[1]/6)
                # gt2pred,pred2gt = None,None
                chamfer_final,precision_final,recall_final,F_score_final = 0,0,0,0
                for batch_index in range(6):
                    if batch_index == 5:
                        gt2pred,pred2gt, _, _ = self.points_set_distance(
                            src=gt_points[:,batch_size_*batch_index:,:],
                            dst=pred_points[:,batch_size_*batch_index:,:],
                            criterion_mode="l2",
                        )
                    else:
                        gt2pred,pred2gt, _, _ = self.points_set_distance(
                            src=gt_points[:,batch_size_*batch_index:batch_size_*(batch_index+1),:],
                            dst=pred_points[:,batch_size_*batch_index:batch_size_*(batch_index+1),:],
                            criterion_mode="l2",
                        )
                    thre1 = 0.1
                    chamfer = torch.mean(gt2pred) + torch.mean(pred2gt)
                    # 4.Precision
                    precision = (pred2gt < thre1).sum() / pred2gt.numel()
                    # 5.Recall
                    recall = (gt2pred < thre1).sum() / gt2pred.numel()
                    # 6.F-score
                    F_score = 2 * precision * recall / (precision + recall + 1e-7)

                    chamfer_final +=chamfer
                    precision_final +=precision
                    recall_final +=recall
                    F_score_final +=F_score
                eval = eval+(
                    (chamfer_final/6).cpu().numpy(),
                    (precision_final/6).cpu().numpy(),
                    (recall_final/6).cpu().numpy(),
                    (F_score_final/6).cpu().numpy()
                )
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
        # print("K_RMSE: ",self.RMSE.evaluate())
        # print("K_Accuracy: ",self.acc.evaluate())
        # self.RMSE.rmse = []
        # self.acc.tp=0
        # self.acc.total = 0
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel",
                "chamfer","precision","recall","F_score"
        ]
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
            ret_metrics = pre_eval_to_metrics(results,self.eval_chamfer)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        if self.eval_chamfer:
            eval_num = 13
        else:
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
        if self.mask_pe:
            eval_results.update(self.MIoUEvaluator.evaluate())
            eval_results.update({'number of iou over 60: ':len(self.iou_over_60_list)/652})
            eval_results.update({'number of iou over 75: ':len(self.iou_over_75_list)/652})
            self.iou_over_60_list = []
            self.iou_over_75_list = []
        return eval_results
