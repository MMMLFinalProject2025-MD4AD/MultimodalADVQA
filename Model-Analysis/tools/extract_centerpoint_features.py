import argparse
import numpy as np
import os
from tqdm import tqdm
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS, DATASETS
from mmdet3d.apis import init_model
from nuscenes.nuscenes import NuScenes
from matplotlib.path import Path
from copy import deepcopy
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ann_pkl_file', type=str, required=True, default='nuscenes_infos_train.pkl')
    parser.add_argument('--train_or_val', type=str, required=True, default='train')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--nusc_root', type=str, required=True)
    parser.add_argument('--token_list', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--max_obj', type=int, default=100)
    parser.add_argument('--extract_only_zero', action='store_true', help='only output all zero .npz')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def rotate_bev_box(center, size, yaw):
    cx, cy = center
    w, h = size
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return (rotation @ corners.T).T + [cx, cy]


def point_in_rotated_box(points, corners):
    poly = Path(corners)
    return poly.contains_points(points)

def debug_plot_bev_mask(corners, mask, H, W, save_path, token=None, idx=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(mask.reshape(H, W), origin='lower', cmap='gray')
    print(f"corners = {corners}")
    plt.plot(*zip(*np.vstack([corners, corners[0]])), color='red')  # box outline
    title = f"Mask with Box Overlay"
    if token and idx is not None:
        title += f"\nToken: {token}, Box #{idx}"
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def mean_pool_feat(bev_feat, box, voxel_size, pc_range, args, debug_token=None, debug_idx=None):
    if args.debug:
        print(f"pc_range = {pc_range}")
        print(f"voxel_size = {voxel_size}")
    x, y, z, dx, dy, dz, yaw = box
    # point_cloud_range and voxel_size must be lists or arrays of length 2
    x_range = pc_range[3] - pc_range[0]  # e.g., 54 - (-54) = 108
    y_range = pc_range[4] - pc_range[1]

    bev_input_W = x_range / voxel_size[0]  # Input resolution along width
    bev_input_H = y_range / voxel_size[1]  # Input resolution along height

    # Extract output BEV shape from feature map
    _, C, bev_H, bev_W = bev_feat.shape

    # Compute downsampling stride
    out_stride_W = bev_input_W / bev_W
    out_stride_H = bev_input_H / bev_H    
    if args.debug:
        print(f"BEV Input Resolution: ({bev_input_W}, {bev_input_H})")
        print(f"BEV Feature Map Size: ({bev_W}, {bev_H})")
        print(f"BEV Output Stride: (W: {out_stride_W}, H: {out_stride_H})")

    cx = (x - pc_range[0]) / (voxel_size[0]*out_stride_W)
    cy = (y - pc_range[1]) / (voxel_size[1]*out_stride_H)
    w = dx / voxel_size[0]
    h = dy / voxel_size[1]

    corners = rotate_bev_box([cx, cy], [w, h], yaw)

    if args.debug:
        print(f"corners = {corners}")
        print(f"bev_feat.shape = {bev_feat.shape}")

    xs = np.arange(bev_W) + 0.5
    ys = np.arange(bev_H) + 0.5
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)

    if args.debug:
        print(f"grid = {grid}")

    mask = point_in_rotated_box(grid, corners)
    if not np.any(mask):
        if args.debug:
            print(f"no pts in mask")
            print(f"no pts in mask for box {debug_idx} in token {debug_token}")
            save_path = f"./debug_bev_masks/{debug_token}_box{debug_idx}.png"
            debug_plot_bev_mask(corners, mask, bev_H, bev_W, save_path, debug_token, debug_idx)        
            input()
        return np.zeros((C,), dtype=np.float32)
    indices = np.where(mask.reshape(bev_H, bev_W))
    ret = bev_feat[0, :, indices[0], indices[1]].mean(dim=1).cpu().numpy()

    if args.debug:
        print(f"indices = {indices}")
        print(f"ret.shape = {ret.shape}")
        input()

    return ret

def fix_input_paths(info_dict, data_root):
    # Fix LIDAR path
    lidar_fname = info_dict['lidar_points']['lidar_path']
    if not lidar_fname.startswith('samples'):
        if '__' in lidar_fname:
            sensor = lidar_fname.split('__')[1]
            lidar_path = os.path.join('samples', sensor, lidar_fname)
            info_dict['lidar_points']['lidar_path'] = os.path.join(data_root, lidar_path)

    # Fix camera image paths
    if 'images' in info_dict:
        for cam_name, cam_info in info_dict['images'].items():  # Fix here
            img_fname = cam_info['img_path']
            if not img_fname.startswith('samples'):
                if '__' in img_fname:
                    sensor = img_fname.split('__')[1]
                    img_path = os.path.join('samples', sensor, img_fname)
                    cam_info['img_path'] = os.path.join(data_root, img_path)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # === Step 1: Build dataset config for NuScenesDataset ===
    if args.train_or_val == 'train':
        dataset_cfg = deepcopy(cfg.train_dataloader.dataset)
    else:
        dataset_cfg = deepcopy(cfg.val_dataloader.dataset)

    if 'dataset' in dataset_cfg:  # unwrap CBGSDataset
        dataset_cfg = dataset_cfg['dataset']

    # Override config safely
    dataset_cfg.update(
        ann_file=args.ann_pkl_file,
        data_root=args.nusc_root,
        test_mode=True,
        pipeline=cfg.test_pipeline,
        filter_empty_gt=False,
    )
    dataset = DATASETS.build(dataset_cfg)
    dataset.full_init()
    pipeline = dataset.pipeline

    # === Step 2: Load .pkl directly for debug ===
    ann_data = load(os.path.join(args.nusc_root, args.ann_pkl_file))
    if args.debug:
        print(f"[DEBUG] Raw ann keys: {ann_data.keys()}")
        print(f"[DEBUG] Total data_list length: {len(ann_data['data_list'])}")
        print(f"[DEBUG] First raw token: {ann_data['data_list'][0]['token']}")

    # === Step 3: Get sample tokens ===
    raw_data_list = ann_data['data_list']
    sample_token_to_idx = {info['token']: i for i, info in enumerate(raw_data_list)}
    if args.debug:
        print(f"[DEBUG] First 5 tokens: {list(sample_token_to_idx.keys())[:5]}")
        print(f"[DEBUG] sample_token_to_idx size: {len(sample_token_to_idx)}")

    # === Step 4: Load QA tokens and check overlap ===
    with open(args.token_list) as f:
        sample_tokens = [line.strip() for line in f.readlines()]
    
    if args.debug:
        print(f"[DEBUG] Loaded {len(sample_tokens)} QA tokens")
        print(f"[DEBUG] First 5 QA tokens: {sample_tokens[:5]}")

    intersect = set(sample_tokens) & set(sample_token_to_idx.keys())

    if args.debug:
        print(f"[DEBUG] Overlapping tokens: {len(intersect)} / {len(sample_tokens)}")
        if len(intersect) == 0:
            print("[WARNING] No overlapping tokens found. Check ann_file content and token list source!")

    model = init_model(cfg, args.checkpoint, device=torch.device('cuda:0'))
    voxel_size = cfg.model.data_preprocessor.voxel_layer.voxel_size
    pc_range = cfg.model.data_preprocessor.voxel_layer.point_cloud_range
    #voxel_size = cfg.voxel_size
    #pc_range = cfg.point_cloud_range

    os.makedirs(args.output_dir, exist_ok=True)
    saved, skipped = 0, 0

    for token in tqdm(sample_tokens):
        if args.extract_only_zero:
            continue

        if token not in sample_token_to_idx:
            skipped += 1
            continue

        idx = sample_token_to_idx[token]
        info = deepcopy(raw_data_list[idx])
        fix_input_paths(info, args.nusc_root)        
        input_dict = pipeline(info)
        # === Sanity Checks ===
        if 'inputs' not in input_dict:
            raise ValueError(f"[ERROR] Missing 'inputs' for token {token}")

        if 'data_samples' not in input_dict:
            raise ValueError(f"[ERROR] Missing 'data_samples' for token {token}")

        if not isinstance(input_dict['inputs'], dict):
            raise TypeError(f"[ERROR] 'inputs' should be a dict, got {type(input_dict['inputs'])} instead")

        if 'points' not in input_dict['inputs']:
            raise ValueError(f"[ERROR] 'points' not found in inputs for token {token}")

        points = input_dict['inputs']['points']
        if points.shape[0] == 0:
            raise ValueError(f"[ERROR] Empty point cloud for token {token}")

        if saved == 0 and skipped == 0:
            print(f"[DEBUG] input_dict example for token {token}:")
            print(f" - inputs keys: {input_dict['inputs'].keys()}")
            print(f" - data_samples keys: {input_dict['data_samples'].__dict__.keys()}")

        points = input_dict['inputs']['points']
        assert isinstance(input_dict['inputs']['points'], torch.Tensor)
        input_dict['inputs']['points'] = [input_dict['inputs']['points']]        
        data = model.data_preprocessor(input_dict, False)
        data['data_samples'] = [data['data_samples']]
        for ds in data['data_samples']:
            ds.set_metainfo({
                'box_type_3d': dataset.box_type_3d,
                'box_mode_3d': dataset.box_mode_3d
            })

        if args.debug:
            print("Voxel size:", voxel_size)
            print("Point cloud range:", pc_range)
        points = input_dict['inputs']['points']

        if args.debug:
            print(f"len(points) = {points[0]}")
            print(f"points = {points[0]}")
            print(f"[DEBUG] points shape: {points[0].shape}")
            print(f"[DEBUG] point min: {points[0].min(dim=0).values}")
            print(f"[DEBUG] point max: {points[0].max(dim=0).values}")
            print(f"[DEBUG] configured pc_range: {pc_range}")        

        voxel_dict = data['inputs']['voxels']

        if args.debug:
            print(f"voxels shape: {voxel_dict['voxels'].shape}")
            print(f"coors shape: {voxel_dict['coors'].shape}")
            print(f"num_points shape: {voxel_dict['num_points'].shape}")
            print(f"max coors: {voxel_dict['coors'].max(dim=0).values}")
            # Get the sparse shape from config
            sparse_shape = cfg.model.pts_middle_encoder.sparse_shape  # [41, 1440, 1440] for example
            _, H_feat, W_feat = sparse_shape
            print(f"sparse_shape = {sparse_shape}")

            # Your voxel coordinates: (N, 4) => (batch_idx, z, y, x)
            coors = data['inputs']['voxels']['coors']
            max_z, max_y, max_x = coors[:, 1].max(), coors[:, 2].max(), coors[:, 3].max()
            print(f"[CHECK] Max coords - Z: {max_z}, Y: {max_y}, X: {max_x}")
            print(f"[CHECK] Sparse shape - Z: {sparse_shape[0]}, Y: {sparse_shape[1]}, X: {sparse_shape[2]}")
            #input()
   
        with torch.no_grad():
            feats = model.extract_feat(data['inputs'], data['data_samples'])
            preds = model.test_step(data)[0]


        if args.debug:
            print(f"[DEBUG] feats type: {type(feats)}")
            print(f"[DEBUG] feats[0] type: {type(feats[0])}")
            print(f"[DEBUG] feats type: {type(feats)}")
            print(f"[DEBUG] feats len: {len(feats)}")
            for i, f in enumerate(feats):
                print(f"[DEBUG] feats[{i}] type: {type(f)}")
                if isinstance(f, torch.Tensor):
                    print(f"  -> feats[{i}].shape = {f.shape}")
                elif isinstance(f, list):
                    print(f"  -> feats[{i}] is a list of length {len(f)}")
                    for j, item in enumerate(f):
                        if isinstance(item, torch.Tensor):
                            print(f"    -> feats[{i}][{j}].shape = {item.shape}")

        bev_feat = feats[1][0]  # [1, C, H, W]
        pred_boxes = preds.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
        pred_labels = preds.pred_instances_3d.labels_3d.cpu().numpy()

        if len(pred_boxes) == 0:
            print(f"[WARN] No predictions for token: {token}")
            skipped += 1
            continue

        pooled_feats, pooled_boxes, pooled_labels = [], [], []
        for i in range(min(len(pred_boxes), args.max_obj)):
            assert len(pred_boxes[i]) == 9, f"Expected 9-dim box, got {len(pred_boxes[i])}"
            f = mean_pool_feat(bev_feat, pred_boxes[i][:7], voxel_size, pc_range, args, debug_token=token, debug_idx=i)

            if args.debug:
                print(f"f.shape = {f.shape}")
            pooled_feats.append(f)
            pooled_boxes.append(pred_boxes[i][:7])
            pooled_labels.append(int(pred_labels[i]))

        while len(pooled_feats) < args.max_obj:
            pooled_feats.append(np.zeros(args.feat_dim, dtype=np.float32))
            pooled_boxes.append(np.zeros(7, dtype=np.float32))
            pooled_labels.append(0)


        if args.debug:
            print(f"[DEBUG] pooled_feats len: {len(pooled_feats)}")
            print(f"[DEBUG] pooled_feats[0].shape: {pooled_feats[0].shape}")
            print(f"[DEBUG] pooled_feats[1].shape: {pooled_feats[1].shape}")
            print(f"[DEBUG] pooled_feats[2].shape: {pooled_feats[2].shape}")

        results = []
        for feat, box, label in zip(pooled_feats, pooled_boxes, pooled_labels):
            results.append({
                'feats': feat,        # shape (512,)
                'box': box,           # shape (7,)
                'label': int(label)   # single int
            })

        np.savez_compressed(
            os.path.join(args.output_dir, f'{token}.npz'),
            results=results
        )            
        '''
        np.savez_compressed(
            os.path.join(args.output_dir, f'{token}.npz'),
            results=[{
                'feats': np.array(pooled_feats, dtype=np.float32),
                'box': np.array(pooled_boxes, dtype=np.float32),
                'label': int(pooled_labels[0])
            }]
        )
        '''
        saved += 1

    if args.extract_only_zero:
        for token in tqdm(sample_tokens):
            results = []
            for i in range(args.max_obj):
                results.append({
                    'feats': np.zeros(args.feat_dim, dtype=np.float32),        # shape (512,)
                    'box': np.zeros(7, dtype=np.float32),           # shape (7,)
                    'label': 0  # single int
                })

            np.savez_compressed(
                os.path.join(args.output_dir, f'{token}.npz'),
                results=results
            )
            saved += 1


    print(f"[RESULT] Saved {saved} .npz files, skipped {skipped} tokens")


if __name__ == '__main__':
    main()
