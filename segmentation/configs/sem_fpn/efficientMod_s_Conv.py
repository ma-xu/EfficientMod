_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='efficientMod_s_Conv',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=\
                '{path-to-pre-trained-checkpoint}',
            ),
        ),
    neck=dict(in_channels=[32, 64, 144, 312]),
    decode_head=dict(num_classes=150))


gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict( policy='CosineAnnealing',
                  warmup='linear',
                  warmup_iters=1000,
                  warmup_ratio=1.0 / 10,
                  by_epoch=False,
                  min_lr_ratio=0)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
device = 'cuda'
