model = dict(
    type='FBA',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='FBAResnetDilated',
            depth=50,
            in_channels=11,
            stem_channels=64,
            base_channels=64,
            conv_cfg=dict(type='ConvWS'),
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU', inplace=True)),
        decoder=dict(
            type='FBADecoder',
            pool_scales=(1, 2, 3, 6),
            in_channels=2048,
            channels=256,
            conv_cfg=dict(type='ConvWS'),
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='LeakyReLU', inplace=False))),
    loss_alpha=dict(type='L1Loss'),
    pretrained='/nfs/Code/mmediting/resnet50-19c8e357.pth'
    # loss_alpha_lap=dict(type='LapLoss'),
    # loss_alpha_grad=dict(type='GradientLoss'),
    # loss_alpha_compo=dict(type='L1CompositionLoss'),
    # loss_fb=dict(type='L1Loss', loss_weight=1),
    # loss_fb_compo=dict(type='FBACompLoss', loss_weight=0.25),
    # loss_fb_lap=dict(type='LapLoss', channels=3, loss_weight=0.25),
    # loss_exclusion=dict(type='ExclLoss', channels=3, loss_weight=0.25)
)
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = './data/adobe_composition-1k/'
bg_dir = './data/coco/train2014'
fg_dirs = [
    f'{data_root}Training_set/Adobe-licensed images/fg_extended',
    f'{data_root}Training_set/Other/fg_extended'
]
alpha_dirs = [
    f'{data_root}Training_set/Adobe-licensed images/alpha',
    f'{data_root}Training_set/Other/alpha'
]
img_norm_cfg = dict(mean=[0.406,0.456, 0.485 ], std=[ 0.225, 0.224, 0.229])
io_backend_cfg = dict(io_backend='memcached',
        server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
        client_cfg='/mnt/lustre/share/memcached_client/client.conf')

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        flag='grayscale',
        use_cache=True),
    dict(
        type='LoadImageFromFile',
        key='fg',
        save_original_img=True,
        use_cache=True),
    dict(type='RandomLoadResizeBg', bg_dir=bg_dir),
    dict(type='CompositeFg', fg_dirs=fg_dirs, alpha_dirs=alpha_dirs),
    dict(type='GenerateTrimap', kernel_size=(3, 25)),
    dict(
        type='CropAroundUnknown',
        unknown_source='trimap',
        keys=['alpha', 'fg', 'bg', 'ori_fg','trimap'],
        crop_sizes=[320, 480, 640]),
    dict(type='RandomJitter'), #bgr
    dict(type='Flip', keys=['alpha', 'fg', 'bg','trimap']),
    dict(
        type='Resize',
        keys=['alpha', 'fg', 'bg', 'ori_fg','trimap'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='PerturbBg'),
    dict(type='MergeFgAndBg'),
    dict(type='TransformTrimap'),
    dict(
        type='RescaleToZeroOne',
        keys=[
            'merged', 'alpha', 'fg', 'bg', 'trimap', 'trimap_transformed',
            'two_channel_trimap'
        ]),
    dict(
        type='Normalize', keys=['merged'], save_original=True, **img_norm_cfg),
    dict(
        type='Collect',
        keys=[
            'merged', 'trimap_transformed', 'two_channel_trimap',
            'merged_unnormalized', 'alpha', 'fg', 'bg', 'ori_fg',
            'two_channel_trimap'
        ],
        meta_keys=[]),
    dict(
        type='ImageToTensor',
        keys=[
            'merged', 'trimap_transformed', 'two_channel_trimap',
            'merged_unnormalized', 'alpha', 'fg', 'bg', 'ori_fg'
        ]),
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='merged',
        save_original_img=True),
    dict(type='Pad', keys=['trimap', 'merged'], ds_factor=8, mode='reflect'),
    dict(type='TransformTrimap'),
    dict(
        type='RescaleToZeroOne',
        keys=['merged', 'trimap', 'ori_merged', 'two_channel_trimap']),
    dict(
        type='Normalize', keys=['merged'], save_original=True, **img_norm_cfg),
    dict(
        type='Collect',
        keys=[
            'merged', 'trimap_transformed', 'two_channel_trimap',
            'merged_unnormalized'
        ],
        meta_keys=[
            'merged_path', 'merged_ori_shape', 'ori_alpha', 'ori_trimap',
            'ori_merged', 'pad'
        ]),
    dict(
        type='ImageToTensor',
        keys=[
            'merged', 'trimap_transformed', 'two_channel_trimap',
            'merged_unnormalized'
        ]),
]
data = dict(
    workers_per_gpu=5,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'training_list_fba.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/nfs/Code/mmediting/tempval.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizers = dict(type='Adam', lr=1e-5, betas=[0.5, 0.999])
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
evaluation = dict(interval=10000, save_image=True)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='fba'))
    ])

# runtime settings
total_iters = 200000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fba_real_4'
load_from = None
resume_from = "/nfs/Code/mmediting/work_dirs/fba_real_4/iter_8000.pth"
workflow = [('train', 1)]
