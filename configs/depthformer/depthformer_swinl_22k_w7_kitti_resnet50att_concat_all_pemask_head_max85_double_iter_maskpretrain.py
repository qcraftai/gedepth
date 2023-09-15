USEPE_FLAGS = True
USE_GT_FILTER_FLAGS = False
USE_ConvexHull_FLAGS = False
USE_Erode_FLAGS = False
USE_Dilate_FLAGS = False
USE_MASK_PE_FLAGS = False
_base_ = [
    '../_base_/models/depthformer_swin.py',
    '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'KITTIDataset'
data_root = 'data/kitti'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (352, 704)
train_pipeline = [
    dict(type='LoadImageFromFile',USEPE=USEPE_FLAGS,
                 USE_GT_FILTER=USE_GT_FILTER_FLAGS,
                 USE_ConvexHull=USE_ConvexHull_FLAGS,
                 USE_Erode=USE_Erode_FLAGS,
                 USE_Dilate=USE_Dilate_FLAGS),
    dict(type='DepthLoadAnnotations'),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=True),
    dict(type='Resize',ratio_range=(0.5,2.0)),
    dict(type='Padding',img_padding_value=(0,0,0),depth_padding_value=255),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 704)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
test_pipeline = [
    dict(type='LoadImageFromFile',USEPE=USEPE_FLAGS,
                 USE_GT_FILTER=False,
                 USE_ConvexHull=False,
                 USE_Erode=False,
                 USE_Dilate=False,
                 USE_MASK_PE=USE_MASK_PE_FLAGS),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1216, 352),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg',
                            'cam_intrinsic')),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='gt_depth',
        depth_scale=256,
        split='kitti_eigen_train.txt',
        pipeline=train_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='gt_depth',
        depth_scale=256,
        split='kitti_eigen_test.txt',
        pipeline=test_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='gt_depth',
        depth_scale=256,
        split='kitti_eigen_test.txt',
        pipeline=test_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80))

model = dict(
    # pretrained="/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/pretrain/swin_large_patch4_window7_224_22k.pth",
    pretrained = None,
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        num_stages=0,
        USEPE=USEPE_FLAGS),
    neck=dict(
        type='HAHIHeteroNeck',
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=256),
        in_channels=[64, 192, 384, 768, 1536],
        out_channels=[64, 192, 384, 768, 1536],
        embedding_dim=512,
        scales=[1, 1, 1, 1, 1]),
    decode_head=dict(
        type='DenseDepthHead',
        act_cfg=dict(type='LeakyReLU', inplace=True),
        in_channels=[64, 192, 384, 768, 1536],
        up_sample_channels=[64, 192, 384, 768, 1536],
        # in_channels=[65, 193, 385, 769, 1537],
        # up_sample_channels=[65, 193, 385, 769, 1537],
        # channels=65,
        channels=64,
        min_depth=1e-3,
        max_depth=80,
    ))
# schedules
# optimizer
max_lr=1e-4
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1600 * 16,
    warmup_ratio=1.0 / 1000,
    min_lr_ratio=1e-8,
    by_epoch=False) # test add by_epoch false
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1600 * 48)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=800, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"), 
                  less_keys=("abs_rel", "rmse"))
# log_config = dict(
#     interval=10,
#     hooks=[
#         dict(type="TextLoggerHook"),
#         dict(
#             type="WandbLoggerHook",
#             init_kwargs=dict(
#                 project="DepthFormer Reproduct",
#                 name="PE Transformer",
#                 tags=[
#                     "8gpus",
#                 ],
#                 entity="mazhuang",
#             ),
#             # temporary
#             interval=1,
#         ),
#     ],
# )
log_config = dict(
    _delete_=True,
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

# find_unused_parameters=True
# resume_from = "/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/work_dirs/depthformer_swinl_22k_w7_kitti_pe_random_scale_resnet50att_debug/best_abs_rel_iter_27200.pth"
load_from = "/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/work_dirs/depthformer_swinl_22k_w7_kitti_pe_random_scale_resnet50att_pemask_depthmask_3_grad_double_iter/best_abs_rel_iter_70400.pth"