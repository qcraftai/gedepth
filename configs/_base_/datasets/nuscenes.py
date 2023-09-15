# dataset settings
dataset_type = 'NUSCENESDataset' #1600*640
data_root = 'data/nuscenes'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (352, 704)
train_pipeline = [
    dict(type='LoadNuScenesImageFromFile'),
    dict(type='DepthLoadNuScenesAnnotations'),
    dict(type='KBCrop', depth=True, height=640, width=1600),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(640, 800)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadNuScenesImageFromFile'),
    dict(type='KBCrop', depth=False, height=640, width=1600),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 640),
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
                            'flip', 'flip_direction', 'img_norm_cfg')),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='depth_gt',
        depth_scale=100,
        split='nuscenes_keyframe_train.txt',
        pipeline=train_pipeline,
        garg_crop=False,
        eigen_crop=False,
        min_depth=1,
        max_depth=100),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='depth_gt',
        depth_scale=100,
        split='nuscenes_keyframe_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=False,
        min_depth=1,
        max_depth=100),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='depth_gt',
        depth_scale=100,
        split='nuscenes_keyframe_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=False,
        min_depth=1,
        max_depth=100))

