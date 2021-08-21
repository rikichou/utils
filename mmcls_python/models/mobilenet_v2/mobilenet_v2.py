_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
load_from = '/media/ruiming/data/workspace/pro/source/mmclassification/pretrained_model/mobilenetv2_imagenet/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'

dataset_type = 'AffectNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=112, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(112, -1), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]


data = dict(
    samples_per_gpu=64,
    workers_per_gpu=12,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.5,
        dataset=dict(
            type=dataset_type,
            data_prefix='data/fer/AffectNet/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_prefix='data/fer/AffectNet/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/fer/AffectNet/val',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

# optimizer
optimizer = dict(type='Adam', lr=0.02)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', gamma=0.92, step=1)
runner = dict(type='EpochBasedRunner', max_epochs=300)