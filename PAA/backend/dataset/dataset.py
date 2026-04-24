import numpy as np
from pathlib import Path

import copy
import tqdm
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
            line = [n, sn] + list(map(str, l.astype(int).tolist()))
            rows.append(','.join(line))

        with open(path, 'w') as f:
            f.write('\n'.join([colnames] + rows))

    def export(self, path: str, drop: list[str] = []):
        """
        path: Directory path of export images and name of meta file
        drop: List of split name to skip
        """
        from shutil import copy2
        from pathlib import Path

        root = Path(path)
        for split in self.splits_name:
            if split in drop:
                continue
            (root / split).mkdir(parents=True, exist_ok=False)

        indices = list(range(self.__len__()))
        indices = [i for i in indices if self.get_split(i) not in drop]

        meta : PAADataset = copy.deepcopy(self)
        meta.labels      = self.labels[indices]
        meta.images      = list()
        meta.splits      = self.splits[indices]

        print(f"[INFO] Start copying data to {root}")
        for i in tqdm.tqdm(range(len(indices))):
            img_path = Path(self.get_image(indices[i]))
            split    = self.get_split(indices[i])
            assert not split in drop, f"found split={split} in {drop}"

            dst_path = root / split / f'{i:08d}{img_path.suffix}'
            meta.images.append(dst_path.as_posix())
            copy2( img_path.as_posix(), dst_path.as_posix())

        meta.save_csv(root.with_suffix('.csv').as_posix())

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
    
    def __radd__(self, other: "PAADataset") -> "PAADataset":
        if other == 0:
            return self
        return self.__add__(other)

    def __add__(self, other: "PAADataset") -> "PAADataset":
        if not isinstance(other, PAADataset):
            return NotImplemented

        base = copy.deepcopy(self)
        base.images.extend(other.images)
        base.labels = np.concat([base.labels, other.labels],
                                axis=0)

        # Update base's split lookup dict
        for name in other.splits_name:
            if not name in base.splits_n2i:
                base.splits_n2i[name] = len(base.splits_n2i)
                base.splits_name.append(name)

        # Convert other.splits to base's index
        cvt_splits = np.array([base.splits_n2i[other.splits_name[split]]
                        for split in other.splits], dtype=int)
        
        # Assign splits
        base.splits = np.concat(
            [base.splits, cvt_splits], axis=0)

        return base

    @property
    def split_names(self) -> list[str]:
        return self.splits_name
    
    @property
    def attriubte_names(self) -> list[str]:
        return self.attributes.list()
    
    @property
    def image_paths(self) -> list[str]:
        return [self.get_image(i) for i in range(len(self.images))]