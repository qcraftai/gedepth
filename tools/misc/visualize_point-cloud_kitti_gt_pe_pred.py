import argparse
import mmcv
import numpy as np
import warnings
import torch
import os
from mmcv import Config, DictAction, mkdir_or_exist
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import cv2
from os import path as osp
from pathlib import Path

from depth.datasets import build_dataloader, build_dataset
from depth.models import build_depther
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
from IPython import embed
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    return args

def build_data_cfg(cfg, cfg_options):
    """Build data config for loading visualization data."""

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    test_data_cfg = cfg.data.test
    show_pipeline = cfg.eval_pipeline
    test_data_cfg['pipeline'] = show_pipeline

    return cfg

def generate_pointcloud_ply(xyz, color, pc_file):
    # how to generate a pointcloud .ply file using xyz and color
    # xyz    ndarray  3,N  float
    # color  ndarray  3,N  uint8
    df = np.zeros((6, xyz.shape[1]))
    df[0] = xyz[0]
    df[1] = xyz[1]
    df[2] = xyz[2]
    df[3] = color[0]
    df[4] = color[1]
    df[5] = color[2]
    float_formatter = lambda x: "%.4f" % x
    points =[]
    for i in df.T:
        points.append("{} {} {} {} {} {} 0\n".format
                      (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                       int(i[3]), int(i[4]), int(i[5])))
    file = open(pc_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()


def load_pe_or_gt(img_path,cam_points,img_tensor_flatten,inv_K,divide,color_channle):
    pe_img = (cv2.imread(img_path,-1)/divide).astype(np.float32)
    # if divide == 100:
    pe_img[pe_img>85] = 0
    y_index, x_index = np.nonzero(pe_img)
    ones = np.ones(len(x_index))
    pix_coords = np.stack([x_index, y_index, ones], axis=0)
    normalize_points = np.dot(inv_K[:3, :3], pix_coords)
    pe_points = (normalize_points * pe_img[y_index, x_index])
    color_pe = np.zeros((pe_points.shape))
    color_pe[color_channle] = 255
    point = np.concatenate((cam_points,pe_points),axis=1)
    color = np.concatenate((img_tensor_flatten,color_pe),axis=1)
    return point, color


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = mmcv.Config.fromfile(args.config)
    cfg = build_data_cfg(cfg, args.cfg_options)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    cfg.model.train_cfg = None
    model = build_depther(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    model.eval()
    # for other models
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    torch.cuda.empty_cache()

    model = MMDataParallel(model, device_ids=[0])

    progress_bar = mmcv.ProgressBar(len(dataset))

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    for idx, input in enumerate(data_loader):

        with torch.no_grad():
            aug_data_dict = {key: [] for key in input}
            for data in [input]:
                for key, val in data.items():
                    aug_data_dict[key].append(val)

            img_file = aug_data_dict['img_metas'][0]._data[0][0]['filename']
            cam_intrinsic = aug_data_dict['img_metas'][0]._data[0][0]['cam_intrinsic']
            img = mmcv.imread(img_file)
            # embed()
            # exit()
            name = osp.splitext(img_file)[0].split('/')[-4] + '_' + osp.splitext(img_file)[0].split('/')[-1]
            output = model(return_loss=False, **aug_data_dict)

        depth = torch.tensor(output[0], dtype=torch.float32)
        h, w = depth.shape[1],depth.shape[2]

        intrinsics = cam_intrinsic

        meshgrid = np.meshgrid(range(w), range(h), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = depth.new_tensor(id_coords)
        pix_coords = torch.cat([id_coords[0].view(-1).unsqueeze(dim=0), id_coords[1].view(-1).unsqueeze(dim=0)], 0)
        ones = torch.ones(1, w * h)
        pix_coords = torch.cat([pix_coords, ones], dim=0) # 3xHW
        
        inv_K = np.array(np.matrix(intrinsics).I)
        inv_K = pix_coords.new_tensor(inv_K)
        cam_points = torch.matmul(inv_K[:3, :3], pix_coords)
        
        depth_flatten = depth.view(-1)
        cam_points = torch.einsum('cn,n->cn', cam_points, depth_flatten)
       
        img_tensor = torch.tensor(img, dtype=torch.uint8)
        img_tensor = img_tensor[:, :, [2, 1, 0]]

        img_tensor_flatten = img_tensor.permute(2, 0, 1).flatten(start_dim=1)

        # pe_path = 'data/kitti/input/'+aug_data_dict['img_metas'][0].data[0][0]['ori_filename'].split('/')[0]+'/'+aug_data_dict['img_metas'][0].data[0][0]['ori_filename'].split('/')[1]+'/pe.png'
        pe_path = 'data/kitti/input/'+aug_data_dict['img_metas'][0].data[0][0]['ori_filename'].split('/')[0]+'/pe/173.png'
        depth_path = 'data/kitti/gt_depth/'+aug_data_dict['img_metas'][0].data[0][0]['ori_filename'].split('/')[1]+'/proj_depth/groundtruth/image_02/'+aug_data_dict['img_metas'][0].data[0][0]['ori_filename'].split('/')[-1]
        # embed()
        # exit()
        point_pe,color_pe = load_pe_or_gt(pe_path,cam_points.numpy(),img_tensor_flatten.numpy(),inv_K,100,0)
        point,color = load_pe_or_gt(depth_path,point_pe,color_pe,inv_K,256,1)
        # embed()
        # exit()
        pe_img = (cv2.imread(pe_path,-1)/100).astype(np.float32)
        pe_img[pe_img>85] = 0
        a = -1.73/pe_img
        pe_slope = np.ones_like(pe_img)
        pe_slope = pe_slope*0.012275292
        pe_offset = 1.73/(pe_slope-a)
        pe_offset[pe_offset<0]=0
        pe_offset[pe_offset>85]=0
        y_index, x_index = np.nonzero(pe_img)
        ones = np.ones(len(x_index))
        pix_coords = np.stack([x_index, y_index, ones], axis=0)
        normalize_points = np.dot(inv_K[:3, :3], pix_coords)
        dynamic_points = (normalize_points * pe_offset[y_index, x_index])
        dynamic_color = np.zeros((dynamic_points.shape))
        dynamic_color[2] = 255

        point = np.concatenate((point,dynamic_points),axis=1)
        color = np.concatenate((color,dynamic_color),axis=1)









        # dynamic_point,color = load_pe_or_gt(depth_path,point_pe,color_pe,inv_K,256,1)
        generate_pointcloud_ply(point, color, os.path.join(args.output_dir, name+'.ply'))
        progress_bar.update()



if __name__ == '__main__':
    main()
