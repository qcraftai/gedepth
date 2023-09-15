_base_ = [
    '../_base_/models/depthformer_swin.py', '../_base_/datasets/nuscenes.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained='/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/pretrain/swin_large_patch4_window7_224_22k.pth', # noqa
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
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
        channels=64,
        min_depth=1e-3,
        max_depth=80,
    ))
# batch size
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        split='nuscenes_keyframe_train.txt',
    ),
    val=dict(
        split='nuscenes_keyframe_test.txt',
    ),
    test=dict(
        split='nuscenes_keyframe_test.txt',
    )
)
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
    warmup_iters=1600 * 8,
    warmup_ratio=1.0 / 1000,
    min_lr_ratio=1e-8,
    by_epoch=False) # test add by_epoch false
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner',  max_epochs=30)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=1)
evaluation = dict(by_epoch=True, 
                  start=0,
                  interval=2, 
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
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook', by_epoch=True)
    ])