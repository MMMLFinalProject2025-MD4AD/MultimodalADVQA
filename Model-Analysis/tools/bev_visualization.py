import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, DATASETS
from mmdet3d.apis import init_model
from mmengine.fileio import load
from copy import deepcopy

def rotate_bev_box(center, size, yaw):
    cx, cy = center
    w, h = size
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return (rotation @ corners.T).T + [cx, cy]

def draw_bev_scene(points, gt_boxes, gt_labels, voxel_size, pc_range, save_path, class_names=None):
    H, W = 512, 512
    bev_img = np.zeros((H, W), dtype=np.float32)
    for x, y, z, *_ in points:
        if not (pc_range[0] <= x < pc_range[3]) or not (pc_range[1] <= y < pc_range[4]):
            continue
        cx = int((x - pc_range[0]) / voxel_size[0] / ((pc_range[3] - pc_range[0]) / W))
        cy = int((y - pc_range[1]) / voxel_size[1] / ((pc_range[4] - pc_range[1]) / H))
        if 0 <= cx < W and 0 <= cy < H:
            bev_img[cy, cx] += 1.0
    if bev_img.max() > 0:
        bev_img = np.log1p(bev_img)
        bev_img = bev_img / bev_img.max()

    plt.figure(figsize=(10, 10))
    plt.imshow(bev_img, cmap='viridis', origin='lower')

    def plot_boxes(boxes, labels, color, prefix):
        for i, box in enumerate(boxes):
            x, y, z, dx, dy, dz, yaw = box[:7]
            cx = (x - pc_range[0]) / voxel_size[0] / ((pc_range[3] - pc_range[0]) / W)
            cy = (y - pc_range[1]) / voxel_size[1] / ((pc_range[4] - pc_range[1]) / H)
            w = dx / voxel_size[0] / ((pc_range[3] - pc_range[0]) / W)
            h = dy / voxel_size[1] / ((pc_range[4] - pc_range[1]) / H)
            corners = rotate_bev_box([cx, cy], [w, h], yaw)
            poly = Polygon(corners, edgecolor=color, fill=False, linewidth=2)
            plt.gca().add_patch(poly)
            label_str = class_names[labels[i]] if class_names is not None else str(labels[i])
            plt.text(cx, cy, f'{prefix}:{label_str}', color=color, fontsize=6, ha='center')

    plot_boxes(gt_boxes, gt_labels, 'lime', 'GT')

    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def fix_input_paths(info_dict, data_root):
    lidar_fname = info_dict['lidar_points']['lidar_path']
    if not lidar_fname.startswith('samples'):
        if '__' in lidar_fname:
            sensor = lidar_fname.split('__')[1]
            lidar_path = os.path.join('samples', sensor, lidar_fname)
        else:
            lidar_path = lidar_fname
        abs_path = os.path.join(data_root, lidar_path)
        if not os.path.exists(abs_path):
            abs_path = os.path.join(data_root, 'samples/LIDAR_TOP', lidar_fname)
        info_dict['lidar_points']['lidar_path'] = abs_path

    if 'images' in info_dict:
        for cam_name, cam_info in info_dict['images'].items():
            img_fname = cam_info['img_path']
            if not img_fname.startswith('samples'):
                if '__' in img_fname:
                    sensor = img_fname.split('__')[1]
                    img_path = os.path.join('samples', sensor, img_fname)
                else:
                    img_path = img_fname
                cam_info['img_path'] = os.path.join(data_root, img_path)    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--ann_pkl_file', type=str, required=True)
    parser.add_argument('--sample_token', type=str, required=True)
    parser.add_argument('--nusc_root', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='bev_visualization')
    parser.add_argument('--train_or_val', type=str, choices=['train', 'val'], default='val')
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    if args.train_or_val == 'train':
        dataset_cfg = deepcopy(cfg.train_dataloader.dataset)
    else:
        dataset_cfg = deepcopy(cfg.val_dataloader.dataset)

    if 'dataset' in dataset_cfg:
        dataset_cfg = dataset_cfg['dataset']
    dataset_cfg.update(
        ann_file=args.ann_pkl_file,
        data_root=args.nusc_root,
        test_mode=False,
        pipeline=cfg.train_pipeline,
        filter_empty_gt=False,
    )

    dataset = DATASETS.build(dataset_cfg)
    dataset.full_init()

    token_to_idx = {dataset.get_data_info(i)['token']: i for i in range(len(dataset))}
    idx = token_to_idx[args.sample_token]
    data_info = dataset.get_data_info(idx)
    #fix_input_paths(data_info, args.nusc_root)
    input_dict = dataset.pipeline(deepcopy(data_info))

    gt_boxes = input_dict['data_samples'].gt_instances_3d.bboxes_3d.tensor.cpu().numpy()
    gt_labels = input_dict['data_samples'].gt_instances_3d.labels_3d.cpu().numpy()
    voxel_size = cfg.model.data_preprocessor.voxel_layer.voxel_size
    pc_range = cfg.model.data_preprocessor.voxel_layer.point_cloud_range
    class_names = dataset.metainfo['classes'] if 'classes' in dataset.metainfo else None

    points = input_dict['inputs']['points'].cpu().numpy()
    save_path = os.path.join(args.output_path, f'{args.sample_token}.png')
    draw_bev_scene(points, gt_boxes, gt_labels, voxel_size, pc_range, save_path, class_names)
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    main()