val_pipeline = [dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(1500, 1500), keep_ratio=False),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs')
                ]

test_pipeline = val_pipeline

data_preprocessor = dict(bgr_to_rgb=True,
                         mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375],
                         size=(224, 224),
                         pad_val=0,
                         seg_pad_val=0,
                         type='SegDataPreProcessor')

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    test_cfg=dict(mode='slide', crop_size=(224, 224), stride=(150, 150)),
    backbone=dict(type='Dinov2Adapter',
                  arch_name='facebook/dinov2-base',
                  pretrained=False,
                  patch_size=14,
                  freeze=True,
                  output_layers=[3, 6, 9, 12],
                  get_cls=False
                  ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=768,  # Adjust based on the number of output channels
        in_index=3,
        # input_transform='multiple_select',
        channels=512,  # Number of channels in intermediate layers
        dilations=(1, 12, 24, 36),
        c1_in_channels=768,  # Channels from the low-level features
        c1_channels=48,  # Reduction for low-level features
        num_classes=2,  # Number of classes in the dataset
        norm_cfg=norm_cfg,
        align_corners=False,
        threshold=0.62,
        out_channels=1,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)

    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,  # Example layer's output channels
        in_index=1,  # Use the output of the 3rd specified layer as auxiliary input
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,  # Number of classes in the dataset
        norm_cfg=norm_cfg,
        align_corners=False,
        out_channels=1,
        threshold=0.62,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)
    )
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

train_cfg = dict(max_iters=10, type='IterBasedTrainLoop', val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(timer=dict(type='IterTimerHook'),
                     logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     sampler_seed=dict(type='DistSamplerSeedHook'),
                     )

custom_hooks = [dict(type='EmptyCacheHook', after_epoch=False, after_iter=True)]
