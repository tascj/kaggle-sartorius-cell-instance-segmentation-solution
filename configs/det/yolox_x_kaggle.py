fp16 = dict(loss_scale=512.)
img_scale = (1536, 1536)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(32, 64),
    random_size_interval=1,
    backbone=dict(type='YOLOPAFPNOfficial', depth=1.33, width=1.25),
    neck=None,
    bbox_head=dict(
        type='YOLOXHeadOfficial',
        num_classes=3,
        width=1.25,
        in_channels=[256, 512, 1024]
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
)

dataset_type = 'CellDataset'
classes = ('shsy5y', 'astro', 'cort')
data_root = '../data/'
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)
    ),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.5, 1.5),
        pad_val=114.0
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dtrain_g0.json',
        img_prefix=data_root + 'train',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))
            ),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ]
    )
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    # persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dval_g0.json',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dval_g0.json',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline,
    ),
)
optimizer = dict(
    type='SGD',
    lr=0.005 / 64,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0)
)
optimizer_config = dict(grad_clip=None)

evaluation = dict(
    interval=1, metric='bbox', classwise=True, proposal_nums=(100, 300, 2000)
)

# num_last_epochs = 5
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=3,
    num_last_epochs=5,
    min_lr_ratio=0.01
)

runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=5)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/yolox_x_livecell/epoch_15.pth'
resume_from = None
workflow = [('train', 1)]
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0002,
        total_iter=500,
        priority=49
    )
]
