cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='NASFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        stack_times=7,
        add_extra_convs='on_input',
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=5),
    bbox_head=dict(
        type='RetinaSepBNHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        num_ins=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='BalancedL1Loss',
            alpha=0.5,
            gamma=1.5,
            beta=0.11,
            loss_weight=1.0)))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/pc/lby/mmdetection-master/data/visdrone2018/annotations/train.json',
        img_prefix='/home/pc/lby/mmdetection-master/data/visdrone2018/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                 'tricycle', 'awning-tricycle', 'bus', 'motor')),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/pc/lby/mmdetection-master/data/visdrone2018/annotations/valid.json',
        img_prefix='/home/pc/lby/mmdetection-master/data/visdrone2018/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                 'tricycle', 'awning-tricycle', 'bus', 'motor')),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/pc/lby/mmdetection-master/data/visdrone2018/annotations/valid.json',
        img_prefix='/home/pc/lby/mmdetection-master/data/visdrone2018/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                 'tricycle', 'awning-tricycle', 'bus', 'motor')))
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[30, 40])
total_epochs = 20
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/pc/lby/mmdetection-master/checkpoints/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth'
resume_from = None
workflow = [('train', 1)]
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
           'tricycle', 'awning-tricycle', 'bus', 'motor')
work_dir = './work_dirs/retinanet_r50_fpn_2x_visdrone2018'
gpu_ids = range(0, 1)
