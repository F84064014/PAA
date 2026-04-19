import torch
import numpy as np
from pathlib import Path

import tqdm
import pickle
import easydict

class PAADataset:
    def __init__(self, path: str) -> None:
        self.path :Path = Path(path).expanduser()
        self.root :Path = self.path.parent

        self.attributes  : list[str]  = None
        self.labels      : np.ndarray = None
        self.images      : list[str]  = None
        self.partition   : dict[np.ndarray] = None

        # Load data
        loader = {
            ".pth": self.load_pth,
            ".pkl": self.load_pkl,
            ".csv": self.load_csv,
        }
        loader[self.path.suffix](self.path.as_posix())

    def load_pth(self, path):
        meta = torch.load(path, weights_only=False)
        self.attributes = meta['attr_name']
        self.labels = meta['label']
        self.images = meta['image_name']
        self.partition = meta['partition']

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            meta: easydict = pickle.load(f)
        self.attributes = meta.attr_name
        self.labels = meta.label
        self.images = meta.image_name
        self.partition = meta.partition

    def load_csv(self, path):
        text = Path(path).read_text()
        lines = text.split('\n')
        self.attributes = lines[0].split(',')[2:]
        self.labels     = np.array([map(int, line.split(',')[2:]) for line in lines[1:]])
        self.images     = [line.split(',')[0] for line in lines[1:]]
        self.partition  = [line.split(',')[1] for line in lines[1:]]

    def validate_path(self):
        for image in self.images:
            if not (self.root / image).exists():
                raise FileNotFoundError((self.root / image).as_posix())
        print("Validating PAADataset done")

    def save_pth(self, path):
        meta = {
            'attr_name':    self.attributes,
            'label':        self.labels,
            'image_name':   self.images,
            'partition':    self.partition
        }
        torch.save(meta, path)

    def save_csv(self, path):
        colnames = ','.join(['file_path'] + self.attributes)

        partition : list[str] = [None] * len(self.labels)
        for split, index in self.partition.items():
            for i in index: partition[i] = str(split)

        rows = list()
        for l, n, p in zip(self.labels, self.images, partition):
            line = [n, p] + list(map(str, l.tolist()))
            rows.append(','.join(line))

        with open(path, 'w') as f:
            f.write('\n'.join([colnames] + rows))

    def get_image(self, index=0):
        return (self.root / self.images[index]).as_posix()
    
    def __len__(self) -> int:
        return len(self.images)
    
    @property
    def attriubte_names(self) -> list[str]:
        return self.attributes