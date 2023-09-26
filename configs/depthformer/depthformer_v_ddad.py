_base_ = [
    '../_base_/models/depthformer_swin.py',
    '../_base_/default_runtime.py'
]
USEPE_FLAG = True
depth_scale = 250
# dataset settings
dataset_type = 'DDADDataset'
data_root = 'data/DDAD'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadDDADImageFromFile',USEPE=USEPE_FLAG,USE_DYNAMIC_PE=True),
    dict(type='DDADDepthLoadAnnotations',USE_DYNAMIC_PE=True),
    dict(type='DDADResize',shape=(384, 640),USE_DYNAMIC_PE=True),
    dict(type='Resize',ratio_range=(0.5,2.0)),
    dict(type='Padding',img_padding_value=(0,0,0),depth_padding_value=255,pe_k=True,ori_h=384,ori_w=640),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.0),
    dict(type='RandomCrop', crop_size=(384, 640)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', depth_scale = depth_scale, **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt','pe_k_gt','height'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',)),
]
test_pipeline = [
    dict(type='LoadDDADImageFromFile',USEPE=USEPE_FLAG,USE_DYNAMIC_PE=True),
    dict(type='DDADResize',shape=(384, 640),depth=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 640),
        flip=False,
        flip_direction='horizontal',
        transforms=[
            dict(type='Normalize', depth_scale=depth_scale,**img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img','height','test'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg',)),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train_dataloader=dict(
        shuffle=True,
        drop_last=True,
        persistent_workers=False),
    val_dataloader=dict(
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(
        shuffle=False,
        persistent_workers=False),
    train=dict(
        type=dataset_type,
        split='splits/ddad_train_split.txt',
        pipeline=train_pipeline,
        min_depth=1e-3,
        max_depth=200,
        cameras = ['CAMERA_%02d' % idx for idx in [1,5,6,9]],
        ),
    val=dict(
        type=dataset_type,
        split='splits/ddad_test_split.txt',
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=200,
        cameras = ['CAMERA_%02d' % idx for idx in [1,5,6,9]],
        ),
    test=dict(
        type=dataset_type,
        split='splits/ddad_test_split.txt',
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=200,
        cameras = ['CAMERA_%02d' % idx for idx in [1,5,6,9]],
        ))



model = dict(
    pretrained=None,
    depth_scale = depth_scale,
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        USEPE = USEPE_FLAG),
    neck=dict(
        type='HAHIHeteroNeck',
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=256),
        in_channels=[64, 192, 384, 768, 1536],
        out_channels=[64, 192, 384, 768, 1536],
        embedding_dim=512,
        scales=[1, 1, 1, 1, 1]),
    pe_mask_neck=dict(
        type='LightPEMASKNeck'
    ),
    decode_head=dict(
        type='DenseDepthHead',
        act_cfg=dict(type='LeakyReLU', inplace=True),
        in_channels=[64, 192, 384, 768, 1536],
        up_sample_channels=[64, 192, 384, 768, 1536],
        channels=64,
        min_depth=1e-3,
        max_depth=200,
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
    min_lr_ratio=1e-8,
    by_epoch=False) # test add by_epoch false
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=38400)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=800)
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=800, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"), 
                  less_keys=("abs_rel", "rmse"))
# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])