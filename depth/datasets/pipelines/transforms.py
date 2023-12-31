import mmcv
import torch
import random
import numpy as np
import os.path as osp
from depth.ops import resize
from ..builder import PIPELINES
from numpy.core.fromnumeric import shape
from mmcv.utils import deprecated_api_warning
import cv2

@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """
    def __init__(self, mean, std, depth_scale=200, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.depth_scale = depth_scale

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        img_pe = results['img']
        if img_pe.shape[-1] == 5:
            img = img_pe[:,:,0:3].copy().astype(np.uint8)
            pe =  img_pe[:,:,3].copy()
            pe[pe>0] = pe[pe>0] / self.depth_scale
            pe_comput = img_pe[:,:,4].copy()
            rgb = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            img_pe = np.concatenate([rgb,pe[:,:,None]],axis=-1)
            results['img'] = np.concatenate([img_pe,pe_comput[:,:,None]],axis=-1)
        else:
            img = img_pe.copy()
            rgb = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            results['img'] = rgb
        results['img_norm_cfg'] = dict(mean=self.mean,
                                       std=self.std,
                                       to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class Padding(object):
    def __init__(self, img_padding_value,depth_padding_value,normals=False,pe_k=False,ori_h=352,ori_w=1216):
        self.img_padding_value = img_padding_value
        self.depth_padding_value = depth_padding_value
        self.normals = normals
        self.pe_k = pe_k
        self.ori_h = ori_h
        self.ori_w = ori_w
    def __call__(self, results):
        image = results['img'].copy()
        depth = results['depth_gt'].copy()
        
        if self.normals:
            normals = results['normals_gt'].copy()
            normals_dtype = normals.dtype
        image_dtype = image.dtype
        depth_dtype = depth.dtype
        img_h,img_w,_ = image.shape
        if img_h < self.ori_h or img_w < self.ori_w:
            new_img = np.zeros((self.ori_h,self.ori_w,5)).astype(image_dtype)
            new_depth = np.zeros((self.ori_h,self.ori_w)).astype(depth_dtype)
            if self.normals:
                new_normals = np.zeros((self.ori_h,self.ori_w,3)).astype(normals_dtype)
            h_off = random.randint(0, self.ori_h-img_h)
            w_off = random.randint(0, self.ori_w-img_w)
            new_img[h_off:h_off + img_h, w_off:w_off + img_w] = image
            new_depth[h_off:h_off + img_h, w_off:w_off + img_w] = depth
            if self.normals:
                new_normals[h_off:h_off + img_h, w_off:w_off + img_w] = normals
            results['img'] = new_img
            results['depth_gt'] = new_depth
            if self.normals:
                results['normals_gt'] = new_normals
            # results["depth_shape"] = results["depth_gt"].shape
            if self.pe_k:
                pe_k_gt = results['pe_k_gt'].copy()
                pe_k_gt_dtype = pe_k_gt.dtype
                new_pe_k = 255 + np.zeros((self.ori_h,self.ori_w)).astype(pe_k_gt_dtype)
                new_pe_k[h_off:h_off + img_h, w_off:w_off + img_w] = pe_k_gt
                results['pe_k_gt'] = new_pe_k
            # for key in results.get('depth_fields', []):
            #     if 'pe_k_gt' is key:
            #         continue
            #     results[key] = new_depth
        return results



@PIPELINES.register_module()
class NYUCrop(object):
    """NYU standard krop when training monocular depth estimation on NYU dataset.

    Args:
        depth (bool): Whether apply NYUCrop on depth map. Default: False.
    """
    def __init__(self, depth=False):
        self.depth = depth
        

    def __call__(self, results):
        """Call function to apply NYUCrop on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Croped results.
        """

        if self.depth:
            depth_cropped = results["depth_gt"][45:472, 43:608]
            results["depth_gt"] = depth_cropped
            results["depth_shape"] = results["depth_gt"].shape

        img_cropped = results["img"][45:472, 43:608, :]
        results["img"] = img_cropped
        results["ori_shape"] = img_cropped.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class KBCrop(object):
    """KB standard krop when training monocular depth estimation on KITTI dataset.

    Args:
        depth (bool): Whether apply KBCrop on depth map. Default: False.
        height (int): Height of input images. Default: 352.
        width (int): Width of input images. Default: 1216.

    """
    def __init__(self, depth=False, submodel=False, height=352, width=1216, normals=False,pe_k=False):
        self.depth = depth
        self.submodel = submodel
        self.height = height
        self.width = width
        self.normals = normals
        self.pe_k = pe_k

    def __call__(self, results):
        """Call function to apply KBCrop on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Croped results.
        """
        # print(results)
        height = results["img_shape"][0]
        width = results["img_shape"][1]
        top_margin = int(height - self.height)
        left_margin = int((width - self.width) / 2)
        
        if self.depth:
            depth_cropped = results["depth_gt"][top_margin:top_margin +
                                                self.height,
                                                left_margin:left_margin +
                                                self.width]
            results["depth_gt"] = depth_cropped
            results["depth_shape"] = results["depth_gt"].shape
        if self.pe_k:
            depth_cropped = results["pe_k_gt"][top_margin:top_margin +
                                                self.height,
                                                left_margin:left_margin +
                                                self.width]
            results["pe_k_gt"] = depth_cropped
        
        img_cropped = results["img"][top_margin:top_margin + self.height,
                                     left_margin:left_margin + self.width, :]
        results["img"] = img_cropped
        results["ori_shape"] = img_cropped.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate the image & depth.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        depth_pad_val (float, optional): Padding value of depth map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """
    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 depth_pad_val=0,
                 center=None,
                 auto_bound=False,
                 normals=False):
        self.prob = prob
        self.normals = normals
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.depth_pad_val = depth_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(results['img'],
                                           angle=degree,
                                           border_value=self.pal_val,
                                           center=self.center,
                                           auto_bound=self.auto_bound)
            if self.normals:
                results['normals_gt'] = mmcv.imrotate(results['normals_gt'],
                                           angle=degree,
                                           border_value=self.pal_val,
                                           center=self.center,
                                           auto_bound=self.auto_bound,
                                           interpolation='nearest')
            # rotate depth
            for key in results.get('depth_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=255 if "pe" in key else self.depth_pad_val,
                    # border_value=self.depth_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'depth_pad_val={self.depth_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & depth.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """
    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal',normals=False):
        self.prob = prob
        self.normals = normals
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, depth estimation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'],
                                         direction=results['flip_direction'])
            if self.normals:
                results['normals_gt'] = mmcv.imflip(results['normals_gt'],
                                         direction=results['flip_direction'])
            # flip depth
            for key in results.get('depth_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """
    def __init__(self, crop_size, normals=False):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.normals = normals

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        if self.normals:
            normals = self.crop(results['normals_gt'], crop_bbox)
            results['normals_gt'] = normals
        # crop depth
        for key in results.get('depth_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
            
        results["depth_shape"] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class ColorAug(object):
    """Color augmentation used in depth estimation

    Args:
        prob (float, optional): The color augmentation probability. Default: None.
        gamma_range(list[int], optional): Gammar range for augmentation. Default: [0.9, 1.1].
        brightness_range(list[int], optional): Brightness range for augmentation. Default: [0.9, 1.1].
        color_range(list[int], optional): Color range for augmentation. Default: [0.9, 1.1].
    """
    def __init__(self,
                 prob=None,
                 gamma_range=[0.9, 1.1],
                 brightness_range=[0.9, 1.1],
                 color_range=[0.9, 1.1]):
        self.prob = prob
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.color_range = color_range
        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        """Call function to apply color augmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly colored results.
        """
        aug = True if np.random.rand() < self.prob else False

        if aug:
            image = results['img'][:,:,0:3]

            # gamma augmentation
            gamma = np.random.uniform(min(*self.gamma_range),
                                      max(*self.gamma_range))
            image_aug = image**gamma

            # brightness augmentation
            brightness = np.random.uniform(min(*self.brightness_range),
                                           max(*self.brightness_range))
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(min(*self.color_range),
                                       max(*self.color_range),
                                       size=3)
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)],
                                   axis=2)
            image_aug *= color_image
            image_aug = np.clip(image_aug, 0, 255)

            results['img'][:,:,0:3] = image_aug

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class Resize(object):
    """Resize images & depth.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 normals=False):
        
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.normals = normals

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long),
                                      max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short),
                                       max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(results['img'],
                                               results['scale'],
                                               return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(results['img'],
                                                  results['scale'],
                                                  return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_depth(self, results):
        """Resize depth estimation map with ``results['scale']``."""
        for key in results.get('depth_fields', []):
            if self.keep_ratio:
                gt_depth = mmcv.imrescale(results[key],
                                          results['scale'],
                                          interpolation='nearest')
            else:
                gt_depth = mmcv.imresize(results[key],
                                         results['scale'],
                                         interpolation='nearest')
            results[key] = gt_depth

            # print(key,': ',results[key].shape)
            
    
    def _resize_normals(self, results):
        """Resize surface normals map with ``results['scale']``."""
        if self.keep_ratio:
            normals_gt = mmcv.imrescale(results['normals_gt'],
                                          results['scale'],
                                          interpolation='nearest')
        else:
            normals_gt = mmcv.imresize(results['normals_gt'],
                                         results['scale'],
                                         interpolation='nearest')
        results['normals_gt'] = normals_gt

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, depth estimation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_depth(results)
        if self.normals:
            self._resize_normals(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class DDADResize(object):
    
    def __init__(self, shape,depth=True,USE_DYNAMIC_PE=False):
        self.shape = shape
        self.depth = depth
        self.USE_DYNAMIC_PE = USE_DYNAMIC_PE
    def __call__(self, results):
        img_pe = results['img']
        if img_pe.shape[-1] == 5:
            img = img_pe[:,:,0:3].copy().astype(np.uint8)
            pe =  img_pe[:,:,3].copy().astype(np.float32)
            pe_comput =  img_pe[:,:,4].copy().astype(np.float32)
            img = cv2.resize(img, dsize=self.shape[::-1],interpolation=cv2.INTER_AREA)
            pe = cv2.resize(pe, dsize=self.shape[::-1],interpolation=cv2.INTER_NEAREST)
            pe_comput = cv2.resize(pe_comput, dsize=self.shape[::-1],interpolation=cv2.INTER_NEAREST)
            img_pe = np.concatenate([img,pe[:,:,None]],axis=-1).astype(np.float32)
            results['img'] = np.concatenate([img_pe,pe_comput[:,:,None]],axis=-1).astype(np.float32)
        else:
            results['img'] = cv2.resize(img_pe, dsize=self.shape[::-1],interpolation=cv2.INTER_AREA)
        if self.depth:
            depth = results['depth_gt']
            h, w = depth.shape
            x = depth.reshape(-1)
            uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
            idx = x > 0
            crd, val = uv[idx], x[idx]
            crd[:, 0] = (crd[:, 0] * (self.shape[0] / h)).astype(np.int32)
            crd[:, 1] = (crd[:, 1] * (self.shape[1] / w)).astype(np.int32)
            idx = (crd[:, 0] < self.shape[0]) & (crd[:, 1] < self.shape[1])
            crd, val = crd[idx], val[idx]
            depth = np.zeros(self.shape)
            depth[crd[:, 0], crd[:, 1]] = val
            results['depth_gt'] = depth
            if self.USE_DYNAMIC_PE:
                pe_k_gt = results['pe_k_gt']
                h, w = pe_k_gt.shape
                x = pe_k_gt.reshape(-1)
                uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
                idx = x > 0
                crd, val = uv[idx], x[idx]
                crd[:, 0] = (crd[:, 0] * (self.shape[0] / h)).astype(np.int32)
                crd[:, 1] = (crd[:, 1] * (self.shape[1] / w)).astype(np.int32)
                idx = (crd[:, 0] < self.shape[0]) & (crd[:, 1] < self.shape[1])
                crd, val = crd[idx], val[idx]
                pe_k_gt = np.zeros(self.shape)
                pe_k_gt[crd[:, 0], crd[:, 1]] = val
                results['pe_k_gt'] = pe_k_gt
        return results


