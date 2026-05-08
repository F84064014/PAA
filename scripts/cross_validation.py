import sys
sys.path.append('.')

from PAA.backend import (
    PAADataset
)
import argparse
import numpy as np
from sklearn.model_selection import KFold

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      type=str, default='data/RealWorld_0508.pth')
    return parser.parse_args()

def get_binary_cv_masks(n_samples, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    masks = []
    
    for _, test_index in kf.split(np.zeros(n_samples)):
        mask = np.zeros(n_samples, dtype=int)
        mask[test_index] = 1
        masks.append(mask)
        
    return masks

if __name__=="__main__":
    args = parse_arg()
    dataset = PAADataset(args.data)
    dataset.drop('ignore')

    train_id = dataset.splits_n2i['train']
    val_id   = dataset.splits_n2i['val']
    
    for cv_i, mask in enumerate(get_binary_cv_masks(len(dataset))):
        dataset.splits = np.empty_like(dataset.splits)
        dataset.splits[mask==1] = val_id
        dataset.splits[mask==0] = train_id
        dataset.save_pth(f"data/RealWorld_0508_cv{cv_i}.pth")
