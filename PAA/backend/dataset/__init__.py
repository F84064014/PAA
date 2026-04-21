from .dataset import PAADataset
from .attributes import Attributes

def load_data(path: str) -> PAADataset:
    return PAADataset(path)