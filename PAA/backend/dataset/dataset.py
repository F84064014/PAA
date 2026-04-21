import numpy as np
from pathlib import Path

import pickle
import easydict
from .attributes import Attributes

class PAADataset:
    def __init__(self, path: str) -> None:
        self.path :Path = Path(path).expanduser()
        self.root :Path = self.path.parent

        self.attributes  : Attributes = None
        self.labels      : np.ndarray = None
        self.images      : list[str]  = None
        self.splits      : np.ndarray = None
        self.splits_name : list[str]  = None
        self.splits_n2i  : dict[str, int] = None

        # Load data
        loader = {
            ".pth": self.load_pth,
            ".pkl": self.load_pkl,
            ".csv": self.load_csv,
        }
        loader[self.path.suffix](self.path.as_posix())

    def load_pth(self, path):
        import torch
        meta = torch.load(path, weights_only=False)
        self.attributes = Attributes(meta['attr_name'])
        self.labels = meta['label']
        self.images = meta['image_name']

        self.splits = np.empty(len(self.labels), dtype=int)
        self.splits_name = list()
        self.splits_n2i  = dict()
        for split, indices in meta['partition'].items():
            self.splits_name.append(split)
            self.splits[indices] = len(self.splits_name)-1
            self.splits_n2i[split] = len(self.splits_n2i)

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            meta: easydict = pickle.load(f)
        self.attributes = Attributes(meta.attr_name)
        self.labels = meta.label
        self.images = meta.image_name

        self.splits = np.empty(len(self.labels), dtype=int)
        self.splits_name = list()
        for split, indices in meta.partition.items():
            self.splits_name.append(split)
            self.splits[indices] = len(self.splits_name)-1

    def load_csv(self, path):
        text = Path(path).read_text()
        lines = text.split('\n')
        self.attributes = Attributes(lines[0].split(',')[2:])
        self.labels     = np.array([list(map(int, line.split(',')[2:])) for line in lines[1:]])

        self.images      = list()
        self.splits_name = list()
        self.splits      = list()
        self.splits_n2i  = dict()
        for line in lines[1:]:

            imn, sn = line.split(',')[:2]
            if not sn in self.splits_n2i:
                self.splits_n2i[sn] = len(self.splits_n2i)
            self.splits.append(self.splits_n2i[sn])
            self.images.append(imn)

        self.splits = np.array(self.splits)
        self.splits_name = [None] * len(self.splits_n2i)
        for k, v in self.splits_n2i.items():
            self.splits_name[v] = k


    def validate_path(self):
        for image in self.images:
            if not (self.root / image).exists():
                raise FileNotFoundError((self.root / image).as_posix())
        print("Validating PAADataset done")

    def save_pth(self, path):
        import torch
        partition = dict()
        for i, n in enumerate(self.splits_name):
            partition[n] = np.where(self.splits==i)[0]

        meta = {
            'attr_name':    self.attributes.list(),
            'label':        self.labels,
            'image_name':   self.images,
            'partition':    partition
        }
        torch.save(meta, path)

    def save_csv(self, path):
        colnames = ','.join(['file_path', 'split'] + self.attributes.list())

        rows = list()
        for l, n, s in zip(self.labels, self.images, self.splits):
            sn = self.splits_name[s]
            line = [n, sn] + list(map(str, l.tolist()))
            rows.append(','.join(line))

        with open(path, 'w') as f:
            f.write('\n'.join([colnames] + rows))

    def get_image(self, index=0) -> str:
        if Path(self.images[index]).exists():
            return self.images[index]
        else:
            return (self.root / self.images[index]).as_posix()
    
    def get_split(self, index=0) -> str:
        return self.splits_name[self.splits[index]]

    def get_label(self, index=0) -> np.ndarray:
        return self.labels[index]

    def set_label(self, index, label: np.ndarray) -> None:
        self.labels[index] = label

    def set_split(self, index, split: str) -> None:
        self.splits[index] = self.splits_n2i[split]

    def append_split(self, split: str) -> None:
        if not split in self.splits_name:
            self.splits_name.append(split)
            self.splits_n2i[split] = len(self.splits_n2i)
        else:
            print(f"[Warning] split {split} already exist")


    def __len__(self) -> int:
        return len(self.images)
    
    @property
    def split_names(self) -> list[str]:
        return self.splits_name
    
    @property
    def attriubte_names(self) -> list[str]:
        return self.attributes.list()