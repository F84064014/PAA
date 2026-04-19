from .dataset import PAADataset

def load_data(path: str) -> PAADataset:
    return PAADataset(path)