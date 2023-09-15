"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

related codes from https://github.com/NVlabs/pacnet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['packernel2d']

from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.modules.utils import _pair
from IPython import embed

try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass


def _neg_idx(idx):
    return None if idx == 0 else -idx


def np_gaussian_2d(width, sigma=-1):
    '''Truncated 2D Gaussian filter'''
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4

    r = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()

    return gaussian_2d


def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output


class GaussKernel2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, channel_wise):
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        bs, ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1
        out_w = (in_w + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff_sq = (cols - feat_0).pow(2)
        if not channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True)
        output = torch.exp(-0.5 * diff_sq)
        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        grad = -0.5 * grad_output * output
        grad_diff = grad.expand_as(cols) * (2 * diff)
        grad_diff[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] -= \
            grad_diff.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

        grad_input = F.fold(grad_diff.contiguous().view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1),
                            (in_h, in_w), ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        return grad_input, None, None, None, None, None


def packernel2d(input, mask=None, kernel_size=0, stride=1, padding=0, output_padding=0, dilation=1,
                kernel_type='gaussian', smooth_kernel_type='none', smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                channel_wise=False, normalize_kernel=False, transposed=False, native_impl=False):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None
    if mask is not None and mask.dtype != input.dtype:
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)

    if transposed:
        in_sz = tuple(int((o - op - 1 - (k - 1) * d + 2 * p) // s) + 1 for (o, k, s, p, op, d) in
                      zip(input.shape[-2:], kernel_size, stride, padding, output_padding, dilation))
    else:
        in_sz = input.shape[-2:]

    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd2col(mask_pattern, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, transposed=transposed)
        if mask is not None:
            mask = nd2col(mask, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                          dilation=dilation, transposed=transposed)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) \
                       / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern

    if transposed:
        stride = _pair(1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

    if native_impl:
        bs, k_ch, in_h, in_w = input.shape

        x = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()

        if smooth_kernel_type == 'none':
            self_idx = kernel_size[0] * kernel_size[1] // 2
            feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
        else:
            smooth_kernel_size = smooth_kernel.shape[2:]
            smooth_padding = (int(padding[0] - (kernel_size[0] - smooth_kernel_size[0]) / 2),
                              int(padding[1] - (kernel_size[1] - smooth_kernel_size[1]) / 2))
            crop = tuple(-1 * np.minimum(0, smooth_padding))
            input_for_kernel_crop = input.view(-1, 1, in_h, in_w)[:, :,
                                    crop[0]:_neg_idx(crop[0]), crop[1]:_neg_idx(crop[1])]
            smoothed = F.conv2d(input_for_kernel_crop, smooth_kernel,
                                stride=stride, padding=tuple(np.maximum(0, smooth_padding)))
            feat_0 = smoothed.view(bs, k_ch, 1, *x.shape[-2:])
        x = x - feat_0
        if kernel_type.find('_asym') >= 0:
            x = F.relu(x, inplace=True)
        # x.pow_(2)  # this causes an autograd issue in pytorch>0.4
        x = x * x
        if not channel_wise:
            x = torch.sum(x, dim=1, keepdim=True)
        if kernel_type == 'gaussian':
            x = torch.exp_(x.mul_(-0.5))  # TODO profiling for identifying the culprit of 5x slow down
            # x = torch.exp(-0.5 * x)
        elif kernel_type.startswith('inv_'):
            epsilon = 1e-4
            x = inv_alpha.view(1, -1, 1, 1, 1) \
                + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
        else:
            raise ValueError()
        output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()
    else:
        assert (smooth_kernel_type == 'none' and
                kernel_type == 'gaussian')
        output = GaussKernel2dFn.apply(input, kernel_size, stride, padding, dilation, channel_wise)

    if mask is not None:
        output = output * mask  # avoid numerical issue on masked positions

    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    if norm is not None:
        empty_mask = (norm == 0)
        # output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output = output / (norm + empty_mask.clone().detach())
        output_mask = (1 - empty_mask) if output_mask else None
    else:
        output_mask = None

    return output, output_mask