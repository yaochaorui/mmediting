# model settings
model = dict(
    type='GCA',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='ResGCAEncoder',
            block='BasicBlock',
            layers=[3, 4, 4, 2],
            in_channels=9,
            with_spectral_norm=True),
        decoder=dict(
            type='ResGCADecoder',
            block='BasicBlockDec',
            layers=[2, 3, 3, 2],
            with_spectral_norm=True)),
    loss_alpha=dict(type='L1Loss'),
    pretrained='open-mmlab://mmedit/res34_en_nomixup')
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = 'data/adobe_composition-1k'
bg_dir = './data/coco/train2014'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale', use_cache=True),
    dict(type='LoadImageFromFile', key='fg', use_cache=True),
    dict(type='RandomLoadResizeBg', bg_dir=bg_dir),
    dict(
        type='CompositeFg',
        fg_dirs=[
            f'{data_root}/Training_set/Adobe-licensed images/fg',
            f'{data_root}/Training_set/Other/fg'
        ],
        alpha_dirs=[
            f'{data_root}/Training_set/Adobe-licensed images/alpha',
            f'{data_root}/Training_set/Other/alpha'
        ]),
    dict(
        type='RandomAffine',
        keys=['alpha', 'fg'],
        degrees=30,
        scale=(0.8, 1.25),
        shear=10,
        flip_ratio=0.5),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
     dict(
        type='CropAroundUnknown',
        keys=['alpha', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(type='RandomJitter'),
    dict(type='Flip', keys=['alpha', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='PerturbBg'),
    dict(type='MergeFgAndBg'),
    dict(type='GenerateTrimap', kernel_size=(3, 25)),
    dict(type='Pad', keys=['trimap', 'merged'],ds_factor=32, mode='reflect'),
    dict(type='TransformTrimap'),
    dict(type='RescaleToZeroOne', keys=[
        'merged',
        'alpha',
        'fg',
        'bg',
        'trimap',
        'trimap_transformed',
        'two_channel_trimap'
    ]),
    dict(type='Normalize', keys=['merged'], save_original=True, **img_norm_cfg),
    dict(type='Collect', keys=['merged', 'alpha', 'trimap_transformed' ,'trimap'], meta_keys=[]),
    dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap_transformed', 'trimap']),
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        flag='grayscale',
        save_original_img=True, use_cache=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        flag='grayscale',
        save_original_img=True, use_cache=True),
    dict(
        type='LoadImageFromFile',
        key='merged',
        channel_order='rgb',
        save_original_img=True, use_cache=True),
    dict(type='Pad', keys=['trimap', 'merged'],ds_factor=32, mode='reflect'),
    dict(type='TransformTrimap'),
    dict(type='RescaleToZeroOne', keys=['merged', 'trimap','ori_merged','two_channel_trimap']),
    dict(type='Normalize', keys=['merged'], save_original=True, **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'trimap'],
        meta_keys=[
            'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha', 'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'trimap']),
    dict(type='FormatTrimap', to_onehot=True),
]
data = dict(
    workers_per_gpu=0,
    train_dataloader=dict(samples_per_gpu=10, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/training_list.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{data_root}/test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizers = dict(type='Adam', lr=4e-4, betas=[0.5, 0.999])
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001)

# checkpoint saving
checkpoint_config = dict(interval=2000, by_epoch=False)
evaluation = dict(interval=10000, save_image=False)
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='gca'))
    ])

# runtime settings
total_iters = 200000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/gca_bkb'
load_from = None
resume_from = None
workflow = [('train', 1)]