import sys
sys.path.append('.')

from PAA.backend import (
    PAADataset, Model
)
import glob
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref',      type=str, default='data/RealWorld_LAST.pth')
    parser.add_argument('-m', '--model',    type=str, default='../PAR/exp/shufflenetv2_1.0_finetune4_3/shufflenetv2_1.0_finetune4_3.onnx')
    parser.add_argument('-i', '--images',   type=str, default='/mnt/d/yt-dlp/collections/*/crops/*')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arg()
    dataset = PAADataset(args.ref)
    model = Model(args.model)

    dataset.images = glob.glob(args.images)
    dataset.labels = (model(dataset.images) > 0.5).astype(int)
    dataset.splits = [0] * len(dataset.images)
    dataset.save_csv("new_dataset.csv")