_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['oneformer3d'])

# model settings
num_instance_classes = 97
num_semantic_classes = 100
num_channels = 32

model = dict(
    type='ScanNet200OneFormer3DSpConv',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.02,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    query_thr=1,  # 0.5 factor
    query_mult=0.5,  # match training config
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='ScanNetQueryDecoder',
        num_layers=6,
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False),
    criterion=dict(
        type='ScanNetUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.5),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='SparseMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)],
                topk=1),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=600,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1, 2]))

# dataset settings
dataset_type = 'ScanNetPPSegDataset_'
data_root = 'data/scannetpp/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

class_names = [
    'wall',
    'ceiling',
    'floor',
    'table',
    'door',
    'ceiling lamp',
    'cabinet',
    'blinds',
    'curtain',
    'chair',
    'storage cabinet',
    'office chair',
    'bookshelf',
    'whiteboard',
    'window',
    'box',
    'window frame',
    'monitor',
    'shelf',
    'doorframe',
    'pipe',
    'heater',
    'kitchen cabinet',
    'sofa',
    'windowsill',
    'bed',
    'shower wall',
    'trash can',
    'book',
    'plant',
    'blanket',
    'tv',
    'computer tower',
    'kitchen counter',
    'refrigerator',
    'jacket',
    'electrical duct',
    'sink',
    'bag',
    'picture',
    'pillow',
    'towel',
    'suitcase',
    'backpack',
    'crate',
    'keyboard',
    'rack',
    'toilet',
    'paper',
    'printer',
    'poster',
    'painting',
    'microwave',
    'board',
    'shoes',
    'socket',
    'bottle',
    'bucket',
    'cushion',
    'basket',
    'shoe rack',
    'telephone',
    'file folder',
    'cloth',
    'blind rail',
    'laptop',
    'plant pot',
    'exhaust fan',
    'cup',
    'coat hanger',
    'light switch',
    'speaker',
    'table lamp',
    'air vent',
    'clothes hanger',
    'kettle',
    'smoke detector',
    'container',
    'power strip',
    'slippers',
    'paper bag',
    'mouse',
    'cutting board',
    'toilet paper',
    'paper towel',
    'pot',
    'clock',
    'pan',
    'tap',
    'jar',
    'soap dispenser',
    'binder',
    'bowl',
    'tissue box',
    'whiteboard eraser',
    'toilet brush',
    'spray bottle',
    'headphones',
    'stapler',
    'marker'
]

color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(
        type='PointSample_',
        num_points=25000),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=color_mean,
        color_std=color_std),
    dict(
        type='AddSuperPointAnnotations',
        num_classes=num_semantic_classes,
        stuff_classes=[0, 1, 2],
        merge_non_stuff_cls=False),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=0.02,
        p=0.5),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(
        type='PointSample_',
        num_points=25000),
    dict(type='PointSegClassMapping'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=color_mean,
                color_std=color_std),
            dict(
                type='AddSuperPointAnnotations',
                num_classes=num_semantic_classes,
                stuff_classes=[0, 1, 2],
                merge_non_stuff_cls=False),
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

# run settings
train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        ann_file='scannetpp_oneformer3d_infos_train.pkl',
        data_root=data_root,
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline,
        ignore_index=num_semantic_classes,
        scene_idxs=None,
        test_mode=False))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='scannetpp_oneformer3d_infos_val.pkl',
        data_root=data_root,
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))
test_dataloader = val_dataloader

label2cat = {i: name for i, name in enumerate(class_names + ['unlabeled'])}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names + ['unlabeled'],
    dataset_name='ScanNetPP')

sem_mapping = list(range(1, num_semantic_classes + 1))
inst_mapping = sem_mapping[3:]

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0, 1, 2],
    thing_class_inds=list(range(3, num_semantic_classes)),
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)

test_evaluator = val_evaluator

optim_wrapper = dict(
    # type='OptimWrapper',
    accumulative_counts=4,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = dict(type='PolyLR', begin=0, end=512, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou'],
        rule='greater'))

load_from = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=512, val_interval=16)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
