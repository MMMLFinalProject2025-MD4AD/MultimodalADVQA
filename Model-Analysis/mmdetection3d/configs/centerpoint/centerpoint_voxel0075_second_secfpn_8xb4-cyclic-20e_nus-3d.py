_base_ = ['./centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-54, -54.8, -5.0, 54, 53.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_prefix = dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP', img='samples/CAM_FRONT')
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            voxel_size=voxel_size, point_cloud_range=point_cloud_range)),
    pts_middle_encoder=dict(sparse_shape=[41, 1440, 1440]),
    pts_bbox_head=dict(
        bbox_coder=dict(
            voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])),
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])))

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
        'lidar_path',
        'token',  # ✅ Add this line
        'sample_idx',
        'box_type_3d',
        'box_mode_3d',
        'lidar2cam',
        'cam2img',
        'img_path',
        'pcd_horizontal_flip',
        'pcd_vertical_flip',
        'pcd_rotation',
        'pcd_scale_factor',
        'pcd_trans',
        'transformation_3d_flow',
        'num_pts_feats',
        'points',
        'gt_bboxes_3d',
        'gt_labels_3d'
    ]
    )
]
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline, metainfo=dict(classes=class_names))))
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, 
    data_root='data/nuscenes/',  # 🔥 This line is critical
    data_prefix=dict(
        pts='samples/LIDAR_TOP',
        sweeps='sweeps/LIDAR_TOP',
        img='samples/CAM_FRONT'  # ✅ must be here
    ),    
    metainfo=dict(classes=class_names)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, 
    data_root='data/nuscenes/',  # 🔥 This line is critical
    data_prefix=dict(
        pts='samples/LIDAR_TOP',
        sweeps='sweeps/LIDAR_TOP',
        img='samples/CAM_FRONT'  # ✅ must be here
    ),    
    metainfo=dict(classes=class_names)))

#-----------------------------Add by Patrick-----------------------------------#
train_cfg = dict(
    max_epochs=20,       # Train for 20 epochs
    #val_interval=0       # Optional: no validation
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,        # Save every epoch
        max_keep_ckpts=-1  # Keep all checkpoints
    ),
    visualization=dict(
        type='Det3DVisualizationHook',
        draw=True,
        show=False,
        interval=1,
        test_out_dir='./work_dirs/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/eval_epoch10/pred_instances_3d/plots',
        vis_task='lidar_det',
        score_thr=0.3,
        img_prefix='data/nuscenes/samples'
    )    
)