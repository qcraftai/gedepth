import cv2
import json
import mmcv
import numpy as np
import os.path as osp
from PIL import Image
from ..builder import PIPELINES
# from IPython import embed
from functools import reduce
import operator
import math
# import open3d as o3d
import os


def gen_rgb(value, minimum=0, maximum=255):
    """Lidar point visualization
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 8 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255-b-r
    return r,g,b

@PIPELINES.register_module()
class LoadKITTICamIntrinsic(object):
    """Load KITTI intrinsic
    """
    def __init__(self,load_surface_normals=False):
        self.load_surface_normals = load_surface_normals

    def generate_surface_normals(self, depth_map, intrinsic):
        '''
        Args,
            intrinsic, ndarray, (4,4)
            depth_map, ndarray, 

        '''
        intri_K = np.eye(4)
        intri_K[0,0:3] = intrinsic[0]
        intri_K[1,0:3] = intrinsic[1]
        intri_K[2,0:3] = intrinsic[2]

        normals_map = np.zeros([depth_map.shape[0],depth_map.shape[1],3]) # for x,y,z
        y_index, x_index = np.nonzero(depth_map)
        ones = np.ones(len(x_index))
        pix_coords = np.stack([x_index, y_index, ones], axis=0) 
        intrinsic_inv = np.linalg.inv(intri_K)
        normalize_points = np.dot(intrinsic_inv[:3, :3], pix_coords) # x,y,z
        points = normalize_points * depth_map[y_index, x_index]  # cm 
        points = points.T # (3,N) -> (N,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100,max_nn=80))
        normals = np.asarray(pcd.normals)
        normals_map[y_index, x_index] = normals
        return normals_map





    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        # raw input
        if 'input' in  results['img_prefix']:
            date = results['filename'].split('/')[-5]
            results['cam_intrinsic'] = results['cam_intrinsic_dict'][date]
            results['cam_intrinsic_for_normal'] = results['cam_intrinsic_dict_for_nromal'][date]

            if self.load_surface_normals:
                results['normals_gt'] = self.generate_surface_normals(results['depth_gt'],results['cam_intrinsic_for_normal'])
            
        # benchmark test
        else:
            temp = results['filename'].replace('benchmark_test', 'benchmark_test_cam')
            cam_file = temp.replace('png', 'txt')
            results['cam_intrinsic'] = np.loadtxt(cam_file).reshape(3, 3).tolist()
        
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class DepthLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 LOAD_DYNAMIC_PE = False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.LOAD_DYNAMIC_PE = LOAD_DYNAMIC_PE
        

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        depth_gt = np.asarray(Image.open(filename),
                              dtype=np.float32) / results['depth_scale']

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape

        results['depth_fields'].append('depth_gt')
        if not self.LOAD_DYNAMIC_PE:
            # pe_k_gt = np.load(filename.replace('.png','.npz').replace('gt_depth','all_lidar_gt_k_img_wo_vl'))['k_img'].astype(np.float32)
            pe_k_gt = np.load(filename.replace('.png','.npz').replace('gt_depth','slope_range_5_5_interval_1'))['k_img'].astype(np.float32)
            pe_k_gt = pe_k_gt+5
            pe_k_gt[pe_k_gt==260] = 255
            pe_k_gt = cv2.resize(pe_k_gt,(depth_gt.shape[1],depth_gt.shape[0]),interpolation = cv2.INTER_NEAREST)
            results['pe_k_gt'] = pe_k_gt.astype(np.float32)
            results['depth_fields'].append('pe_k_gt')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class DisparityLoadAnnotations(object):
    """Load annotations for depth estimation.
    It's only for the cityscape dataset. TODO: more general.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        if results.get('camera_prefix', None) is not None:
            camera_filename = osp.join(results['camera_prefix'],
                                       results['cam_info']['cam_info'])
        else:
            camera_filename = results['cam_info']['cam_info']

        with open(camera_filename) as f:
            camera = json.load(f)
        baseline = camera['extrinsic']['baseline']
        focal_length = camera['intrinsic']['fx']

        disparity = (np.asarray(Image.open(filename), dtype=np.float32) -
                     1.) / results['depth_scale']
        NaN = disparity <= 0

        disparity[NaN] = 1
        depth_map = baseline * focal_length / disparity
        depth_map[NaN] = 0

        results['depth_gt'] = depth_map
        results['depth_ori_shape'] = depth_map.shape

        results['depth_fields'].append('depth_gt')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 USEPE=False,
                 USE_GT_FILTER=False,
                 USE_ConvexHull=False,
                 USE_Erode=False,
                 USE_Dilate=False,
                 USE_SubModel=False,
                 USE_Ignore=False,
                 USE_MASK_PE=False,
                 USE_PARAM_PE=False,
                 LOAD_DYNAMIC_PE = False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.USEPE = USEPE
        self.USE_GT_FILTER = USE_GT_FILTER
        self.USE_ConvexHull = USE_ConvexHull
        self.USE_Erode = USE_Erode
        self.USE_Dilate = USE_Dilate
        self.USE_SubModel=USE_SubModel
        self.USE_Ignore=USE_Ignore
        self.USE_MASK_PE = USE_MASK_PE
        self.USE_PARAM_PE = USE_PARAM_PE
        self.LOAD_DYNAMIC_PE = LOAD_DYNAMIC_PE

    def pe_erode(self,pe_mask,kernel_size=(9,9),iterations=1):
        """Call functions to erode the pe mask.

        Args:
            pe_mask (numpy): PE Mask
            kernel_size (size,size): erode kernel size

        Returns:
            numpy: Binary mask of after erode pe mask
        """
        kernel = np.ones(kernel_size, 'uint8')
        return cv2.erode(pe_mask, kernel, iterations=iterations)

    def pe_dilate(self,pe_mask,kernel_size=(7,7),iterations=1):
        """Call functions to dilate the pe mask.

        Args:
            pe_mask (numpy): PE Mask
            kernel_size (size,size): dilate kernel size

        Returns:
            numpy: Binary mask of after dilate pe mask
        """
        kernel = np.ones(kernel_size, 'uint8')
        return cv2.dilate(pe_mask, kernel, iterations=iterations)


    def pe_convexhull(self,pe_mask):
        """Call functions to get convex hull of pe.

        Args:
            pe_mask (numpy): PE Mask

        Returns:
            numpy(float32): Convex hull of correct pe
        """
        pe_mask_convexhull = np.zeros_like(pe_mask).astype(np.uint8)
        contours,_ = cv2.findContours(pe_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) > 1:
               cv2.fillConvexPoly(pe_mask_convexhull, cv2.convexHull(contour,clockwise=True,returnPoints=True), (1))
        return pe_mask_convexhull

    def filter_pe(self,pe_mask,pe_depth):
        """Call functions to filter pe value by pe mask.

        Args:
            pe_mask (numpy): PE Mask
            pe_depth (numpy): PE

        Returns:
            numpy(float32): Correct pe
        """
        cv2.imwrite('./pe_mask_dilate777.png',pe_mask*255)
        return pe_mask*pe_depth

    def get_filtered_pe_mask_by_gt(self,results,pe_depth):
        """Call functions to get filtered pe mask by depth gt.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
            pe_depth (numpy): PE

        Returns:
            numpy(uint8): Binary mask of correct pe
        """
        gt_depth = (cv2.imread('data/kitti/gt_depth/'+results['ori_filename'].split('/')[1]+'/proj_depth/groundtruth/image_02/'+results['ori_filename'].split('/')[-1],-1)/256).astype(np.float32)
        gt_err = np.divide(abs(pe_depth-gt_depth),gt_depth,out=np.zeros_like(gt_depth),where=gt_depth!=0)
        gt_mask_err = gt_err<=0.03
        gt_mask = np.logical_and(gt_mask_err,gt_depth)
        return gt_mask.astype(np.uint8)

    def concat_pe_rgb(self,pe_depth,img):
        """Call functions to concat pe and image.

        Args:
            pe_depth (numpy): PE
            img (numpy): RGB

        Returns:
            numpy: concated image,RGBPE=(H,W,4)
        """
        return np.concatenate((img,np.expand_dims(pe_depth, -1)), axis=-1)


    def load_pe_comput(self,results):
        """Call functions to load pe.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            numpy: pe
        """
        pe_depth = np.load('data/kitti/input/'+results['ori_filename'].split('/')[0]+'/pe/pe_165.npy').astype(np.float32)
        if self.LOAD_DYNAMIC_PE:
            filename = osp.join(results['depth_prefix'],results['ann_info']['depth_map'])
            pe_k_gt = np.load(filename.replace('.png','.npz').replace('gt_depth','road_mask_gt_k_img_new'))['k_img'].astype(np.float32)
            pe_k_gt = cv2.resize(pe_k_gt,(pe_depth.shape[1],pe_depth.shape[0]))
            pe_slope_k = np.tan(np.deg2rad(pe_k_gt))
            a = -1.65/pe_depth
            pe_offset = -1.65/((a-pe_slope_k)+1e-8)
            pe_offset[pe_offset>85] = 0
            pe_offset[pe_offset<0] = 0
            return pe_offset
        return pe_depth

    def load_pe(self,results):
        """Call functions to load pe.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            numpy: pe
        """  
        pe_depth = np.load('data/kitti/input/'+results['ori_filename'].split('/')[0]+'/pe/pe_165.npy').astype(np.float32)

        # Filter the non-valid value
        pe_depth[pe_depth>200] = 0
        pe_depth[pe_depth<0] = 0
        # pe_depth[pe_depth<0.001] = 0.001
        return pe_depth    

    def get_ignore_pemask(self,results,pe_mask):
        pe_depth = (cv2.imread('data/kitti/input/'+results['ori_filename'].split('/')[0]+'/'+results['ori_filename'].split('/')[1]+'/pe.png',-1)/100).astype(np.float32)
        pe_depth[pe_depth>85] = 0
        gt_depth = (cv2.imread('data/kitti/gt_depth/'+results['ori_filename'].split('/')[1]+'/proj_depth/groundtruth/image_02/'+results['ori_filename'].split('/')[-1],-1)/256).astype(np.float32)
        gt_err = np.divide(abs(pe_depth-gt_depth),gt_depth,out=np.zeros_like(gt_depth),where=gt_depth!=0)
        gt_mask_err = gt_err<=0.05
        gt_mask = np.logical_and(gt_mask_err,gt_depth)
        un = abs(pe_mask-gt_mask)
        pe_mask[un==1] = 2
        return pe_mask
    
    def create_equal_ratio_points(self,points, ratio, gravity_point):
        if len(points) <= 2 or not gravity_point:
            return list()

        new_points = list()
        length = len(points)

        for i in range(length):
            vector_x = points[i][0] - gravity_point[0]
            vector_y = points[i][1] - gravity_point[1]
            new_point_x = ratio * vector_x + gravity_point[0]
            new_point_y = ratio * vector_y + gravity_point[1]
            new_point = (int(new_point_x), int(new_point_y))
            new_points.append(new_point)
        return new_points

    def auto_canny(self,image, sigma=0.333):
        # v = 1
        # print(v,'66666666666666666666666666666')
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))
        image = (image*255).astype(np.uint8)
        edged = cv2.Canny(image,0,255)
        return edged

    def point2area(self,points, img, color):
        # img = np.zeros_like(img)
        res = cv2.fillPoly(img, [np.array(points)] ,color)
        return res
    def get_mask_pe(self,pe_depth,results):
        # pe_mask_val = (cv2.imread('/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/val_pemaskgt_abs5/'+results['ori_filename'].split('/')[0]+'_'+results['ori_filename'].split('/')[1]+'_image_02_data_'+results['ori_filename'].split('/')[-1],-1)/255).astype(np.float32)
        
        # pe_mask_val = (cv2.imread('/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/work_dirs_backup/maskpe_baseline_CE_Ignore_b32_val/kitti_val/'+results['ori_filename'].split('/')[0]+'_'+results['ori_filename'].split('/')[1]+'_image_02_data_'+results['ori_filename'].split('/')[-1],-1)/255).astype(np.float32)        
        # pe_mask_val = (cv2.imread('/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_weight_backup/kitti_whole_4800_519/'+results['ori_filename'].split('/')[0]+'_'+results['ori_filename'].split('/')[1]+'_image_02_data_'+results['ori_filename'].split('/')[-1],-1)/255).astype(np.float32)
        
        # pe_mask_val = (cv2.imread('/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_Ignore_HRnet/kitti_val/'+results['ori_filename'].split('/')[0]+'_'+results['ori_filename'].split('/')[1]+'_image_02_data_'+results['ori_filename'].split('/')[-1],-1)/255).astype(np.float32)        
        pe_mask_val = (cv2.imread('/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_weight_backup/kitti_val_11200_518/'+results['ori_filename'].split('/')[0]+'_'+results['ori_filename'].split('/')[1]+'_image_02_data_'+results['ori_filename'].split('/')[-1],-1)/255).astype(np.float32)        
        height = pe_depth.shape[0]
        width = pe_depth.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        temp_pe_mask = np.zeros_like(pe_depth).astype(np.uint8)
        temp_pe_mask[top_margin: top_margin + 352, left_margin: left_margin + 1216] = pe_mask_val
        new_mask = np.zeros_like(pe_depth)
        return pe_depth*new_mask

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        # Add PE to channel 4
        if self.USEPE:
            pe_depth = self.load_pe(results)
            pe_depth_comput = self.load_pe_comput(results)
            
            # if self.USE_GT_FILTER:
            #     pe_mask = self.get_filtered_pe_mask_by_gt(results,pe_depth) 
            #     if self.USE_Dilate:
            #         pe_mask = self.pe_dilate(pe_mask) 
            #     if self.USE_ConvexHull:
            #         pe_mask = self.pe_convexhull(pe_mask)              
            #     if self.USE_Erode:
            #         pe_mask = self.pe_erode(pe_mask)
            #     if self.USE_SubModel:
            #         pass
            #     else:
            #         cv2.imwrite('/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/val_pemaskgt_abs5/'+results['ori_filename'].split('/')[0]+'_'+results['ori_filename'].split('/')[1]+'_image_02_data_'+results['ori_filename'].split('/')[-1],pe_mask*255)
            #         pe_depth = self.filter_pe(pe_mask,pe_depth)
            #         img = self.concat_pe_rgb(pe_depth,img)
            # if self.USE_SubModel:
            #     if self.USE_Ignore:
            #         pe_mask = self.get_ignore_pemask(results,pe_mask)
            #     # vis_pe_mask = np.zeros((pe_mask.shape[0],pe_mask.shape[1],3)).astype(np.uint8)
            #     # vis_pe_mask[pe_mask==2] = [0,0,255]
            #     # vis_pe_mask[pe_mask==1] = [255,255,255]
            #     # cv2.imwrite('./pe_mask777.png',vis_pe_mask)
            #     # results['pe_gt'] = pe_mask
            #     # results['depth_fields'].append('pe_gt')
            #     # img = self.concat_pe_rgb(pe_depth,img)
            # if self.USE_MASK_PE:
            #     pe_depth = self.get_mask_pe(pe_depth,results)
            #     img = self.concat_pe_rgb(pe_depth,img)
            # if self.USE_PARAM_PE:
            #     img = self.concat_pe_rgb(pe_depth,img)
            # # cv2.imwrite('./demo_vis.png',pe_mask*255)
            img = self.concat_pe_rgb(pe_depth,img)
            img = self.concat_pe_rgb(pe_depth_comput,img)
            results['pe_ori_point'] = pe_depth_comput[-1,-1]
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



#### seg
#    1. 调研GT训练的模型对PE的灵敏程度
#    2. 分割网络与GT计算IOU
#    3. 高iou单独验证
#    4. 



#### depth


@PIPELINES.register_module()
class LoadNuScenesImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 USEPE=False,
                 USE_GT_FILTER=False,
                 USE_ConvexHull=False,
                 USE_Erode=False,
                 USE_Dilate=False,
                 USE_SubModel=False,
                 USE_Ignore=False,
                 USE_MASK_PE=False,
                 USE_PARAM_PE=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.USEPE = USEPE
        self.USE_GT_FILTER = USE_GT_FILTER
        self.USE_ConvexHull = USE_ConvexHull
        self.USE_Erode = USE_Erode
        self.USE_Dilate = USE_Dilate
        self.USE_SubModel=USE_SubModel
        self.USE_Ignore=USE_Ignore
        self.USE_MASK_PE = USE_MASK_PE
        self.USE_PARAM_PE = USE_PARAM_PE

    def concat_pe_rgb(self,pe_depth,img):
        """Call functions to concat pe and image.

        Args:
            pe_depth (numpy): PE
            img (numpy): RGB

        Returns:
            numpy: concated image,RGBPE=(H,W,4)
        """
        return np.concatenate((img,np.expand_dims(pe_depth, -1)), axis=-1)


    def load_pe(self,results):
        """Call functions to load pe.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            numpy: pe
        """
        pe_depth = (cv2.imread('data/kitti/input/'+results['ori_filename'].split('/')[0]+'/'+results['ori_filename'].split('/')[1]+'/pe.png',-1)/100).astype(np.float32)
        pe_depth[pe_depth>85] = 0
        return pe_depth    

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        # Add PE to channel 4
        if self.USEPE:
            pe_depth = self.load_pe(results)
            img = self.concat_pe_rgb(pe_depth,img)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class DepthLoadNuScenesAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        depth_gt = np.asarray(Image.open(filename),
                              dtype=np.float32) / results['depth_scale']

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape

        results['depth_fields'].append('depth_gt')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



@PIPELINES.register_module()
class DDADDepthLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self, USE_DYNAMIC_PE = False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.USE_DYNAMIC_PE = USE_DYNAMIC_PE
        

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = results['ann_info']['depth_map']

        # depth_gt = np.asarray(Image.open(filename),
        #                       dtype=np.float32) / results['depth_scale']
        depth_gt = np.load(filename)['depth']

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape
        results['depth_fields'].append('depth_gt')

        if self.USE_DYNAMIC_PE:
            pe_k_gt = np.load(filename.replace('depth_val','depth').replace('.npz','_slope_public_debug.npz'))['k_img'].astype(np.float32)
            valid_mask = pe_k_gt==255
            pe_k_gt = pe_k_gt+5
            pe_k_gt[valid_mask] = 255
            results['pe_k_gt'] = pe_k_gt.astype(np.float32)
            results['depth_fields'].append('pe_k_gt')

        # print('depth_gt_ori: ',depth_gt.shape)
        # print('k_ori: ',pe_k_gt.shape)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadDDADImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 USEPE=False,
                 USE_DYNAMIC_PE=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.USEPE = USEPE
        self.USE_DYNAMIC_PE = USE_DYNAMIC_PE

    def concat_pe_rgb(self,pe_depth,img):
        """Call functions to concat pe and image.

        Args:
            pe_depth (numpy): PE
            img (numpy): RGB

        Returns:
            numpy: concated image,RGBPE=(H,W,4)
        """
        return np.concatenate((img,np.expand_dims(pe_depth, -1)), axis=-1)

    def load_pe(self,results):
        """Call functions to load pe.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            numpy: pe
        """
        filename = results['ann_info']['depth_map'].split('/')[-2]
        pe_path = os.path.join('data/DDAD/pe_public_debug',filename+'/ddad_pe.npz')
        pe_depth = np.load(pe_path)['pe']
        pe_depth[pe_depth>250] = 0
        pe_depth[pe_depth<0] = 0
        return pe_depth

    def load_pe_comput(self,results):
        """Call functions to load pe.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            numpy: pe
        """
        # # filename = results['ann_info']['depth_map'].split('/')[-2]
        # # pe_path = os.path.join('/home/mazhuang/workspace/PE/data/ddad/ddad_train_val/pe',filename+'/ddad_pe.npz')
        # filename = results['ann_info']['depth_map'].replace('.npz','_pe.npz')
        # pe_depth = np.load(filename)['pe']
        filename = results['ann_info']['depth_map'].split('/')[-2]
        pe_path = os.path.join('data/DDAD/pe_public_debug',filename+'/ddad_pe.npz')
        pe_depth = np.load(pe_path)['pe']

        # pe_depth[np.isnan(pe_depth)]=0
        # pe_depth[np.isinf(pe_depth)]=500

        return pe_depth

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        # Add PE to channel 4
        if self.USEPE:
            pe_depth = self.load_pe(results)
            img = self.concat_pe_rgb(pe_depth,img)
            if self.USE_DYNAMIC_PE:
                pe_depth_comput = self.load_pe_comput(results)
                img = self.concat_pe_rgb(pe_depth_comput,img)
                if 'CAMERA_01' in filename:
                    h = 1.56
                elif 'CAMERA_05' in filename:
                    h = 1.57
                elif 'CAMERA_06' in filename:
                    h = 1.53
                elif 'CAMERA_09' in filename:
                    h = 1.53 
                results['height'] = h
                results['test'] = 0
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        # print('img: ',img.shape)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



@PIPELINES.register_module()
class LoadDDADCamIntrinsic(object):
    """Load KITTI intrinsic
    """
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        # raw input
        date = results['filename'].split('/')[-2]
        results['cam_intrinsic'] = results['cam_intrinsic_dict'][date]
        
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
