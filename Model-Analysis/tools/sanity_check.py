import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True, default='../../Data/NuScenes-QA/data/features/CenterPoint_epoch1_ft512_train')
    return parser.parse_args()

args = parse_args()
count = 0
for fname in os.listdir(args.feat_dir):
    if not fname.endswith('.npz'):
        continue
    data = np.load(os.path.join(args.feat_dir, fname), allow_pickle=True)['results'][0]
    #feats = data['feats']
    feats = data['box']
    print(f"count={count+1}, fname = {fname}")
    #if feats.shape != (512,):
    if feats.shape != (7,):
        print(f"[ERROR] {fname} has wrong shape: {feats.shape}")
        input()
    count += 1