#!/bin/bash

# Activate your conda environment if needed
# source activate mmdet3d

cuda_device=$1

config="./configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
work_dir="./work_dirs/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d"

#for epoch in $(seq 1 20); do
for epoch in $(seq 10 10); do
    ckpt="${work_dir}/epoch_${epoch}.pth"
    eval_prefix="${work_dir}/eval_epoch${epoch}"
    echo "Running evaluation for epoch ${epoch}..."
    CUDA_VISIBLE_DEVICES=$cuda_device python ./tools/test.py "$config" "$ckpt" --launcher none --cfg-options test_evaluator.jsonfile_prefix="$eval_prefix"
    python ../tools/print_metric_from_json.py --data-root "/data/Datasets/NuScenes-QA/data/nuScenes" --json "${eval_prefix}/pred_instances_3d/results_nusc.json" --out "${eval_prefix}/pred_instances_3d/summary.txt"
done
