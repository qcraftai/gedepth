_base_ = [ '../_base_/models/ocrnet_hr18.py',
    '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py'
]
USEPE_FLAGS = True
USE_GT_FILTER_FLAGS = False
USE_ConvexHull_FLAGS = False
USE_Erode_FLAGS = False
USE_Dilate_FLAGS = False
USE_SubModel_FLAGS = True
USE_Ignore_FLAGS = False
dataset_type = 'KITTIDataset'
data_root = 'data/kitti'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size= (352, 704)
train_pipeline = [
    dict(type='LoadImageFromFile',USEPE=USEPE_FLAGS,
                 USE_GT_FILTER=USE_GT_FILTER_FLAGS,
                 USE_ConvexHull=USE_ConvexHull_FLAGS,
                 USE_Erode=USE_Erode_FLAGS,
                 USE_Dilate=USE_Dilate_FLAGS,
                 USE_SubModel=USE_SubModel_FLAGS,
                 USE_Ignore=USE_Ignore_FLAGS),
    dict(type='DepthLoadAnnotations'),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=True,submodel=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 704)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt','pe_gt'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
test_pipeline = [
    dict(type='LoadImageFromFile',USEPE=USEPE_FLAGS,
                 USE_GT_FILTER=USE_GT_FILTER_FLAGS,
                 USE_ConvexHull=USE_ConvexHull_FLAGS,
                 USE_Erode=USE_Erode_FLAGS,
                 USE_Dilate=USE_Dilate_FLAGS,
                 USE_SubModel=USE_SubModel_FLAGS,
                 USE_Ignore=USE_Ignore_FLAGS),
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
    samples_per_gpu=14,
    workers_per_gpu=8,
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
        max_depth=80,
        mask_pe=True),
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
        max_depth=80,
        mask_pe=True),
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
        max_depth=80,
        mask_pe=True))
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)),
        ),
    ),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='RMILoss', num_classes=2)
            # loss_decode=dict(
            #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.2,1.0])
        ),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='RMILoss', num_classes=2))
    ]
)


# schedules
# optimizer
max_lr=7e-4
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
    warmup_iters=1600 * 8,
    warmup_ratio=1.0 / 1000,
    min_lr_ratio=1e-8,
    by_epoch=False) # test add by_epoch false
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1600 * 24)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=48, interval=800)
# runner = dict(type='IterBasedRunner', max_iters=10)
# checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1)
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=800, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3","mIoU"), 
                  less_keys=("abs_rel", "rmse"))

log_config = dict(
    _delete_=True,
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

# log_config = dict(
#     interval=10,
#     hooks=[
#         dict(type="TextLoggerHook"),
#         dict(
#             type="WandbLoggerHook",
#             init_kwargs=dict(
#                 project="Learn PE Mask",
#                 name="maskpe_baseline_CE_Ignore_weight",
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