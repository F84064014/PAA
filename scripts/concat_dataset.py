import sys
sys.path.append('.')

from PAA.backend import (
    PAADataset
)

if __name__=="__main__":
    paths = sys.argv[1:]
    
    datasets : list[PAADataset] = []
    for path in paths:
        datasets.append(PAADataset(path))
    sum(datasets).save_pth("merge.pth")