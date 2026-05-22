import sys
sys.path.append('.')

from PIL import Image
import imagehash
from pathlib import Path

from PAA.backend import (
    PAADataset
)
import numpy as np

def find_similar_images(image_paths, threshold=5):
    """
    image_paths: List[str] - 圖片路徑列表
    threshold: int - 相似度門檻。0 表示完全相同，數值越大越寬鬆（通常 5 以內非常相似）
    """
    hashes = {}
    duplicates = []
    duplicates_index = []
    
    for i, path_str in enumerate(image_paths):
        path = Path(path_str)
        if not path.exists():
            print(f"跳過：找不到檔案 {path_str}")
            continue
            
        try:
            with Image.open(path) as img:
                # 使用 pHash (感知雜湊)，對縮放、色彩微調不敏感
                h = imagehash.phash(img)
                
                is_duplicate = False
                for existing_path, existing_hash in hashes.items():
                    # 計算兩個雜湊值之間的差異
                    if h - existing_hash <= threshold:
                        print(f"發現相似：\n  1. {existing_path}\n  2. {path_str} (距離: {h - existing_hash})")
                        duplicates_index.append(i)
                        duplicates.append(path_str)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    hashes[path_str] = h
        except Exception as e:
            print(f"無法處理圖片 {path_str}: {e}")

    return duplicates, duplicates_index

# Rmove duplicate data
dataset = PAADataset("data/RealWorld_0507.pth")
images = dataset.image_paths

files, index = find_similar_images(images, threshold=5)
dataset.drop(np.array(index))
dataset.save_pth('data/RealWorld_0507_trim.pth')