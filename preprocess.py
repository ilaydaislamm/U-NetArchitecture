import json

import cv2
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset
from torchvision.transforms import v2
from PIL import Image
from torchvision import tv_tensors

raw_path = Path(r"C:\Users\omen\PycharmProjects\UNetArchitecture\stage1_test")


import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import albumentations as A









def create_weight_maps(train_root, out_dir, w0=10, sigma=5, overwrite=False):
    train_root = Path(train_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())

    for sample_dir in tqdm(sample_dirs):
        out_file = out_dir / f"{sample_dir.name}.npy"
        if out_file.exists() and not overwrite:
            continue

        mask_files = sorted((sample_dir / "masks").glob("*.png"))
        if len(mask_files) == 0:
            continue

        H, W = np.array(Image.open(mask_files[0]).convert("L")).shape

        foreground = np.zeros((H, W), dtype=bool)

        # d1, d2: şimdiye kadar görülen en yakın ve ikinci en yakın nesneye uzaklık
        d1 = np.full((H, W), np.inf, dtype=np.float32)
        d2 = np.full((H, W), np.inf, dtype=np.float32)

        for f in mask_files:

            m = np.array(Image.open(f).convert("L")) > 127
            foreground |= m

            d = distance_transform_edt(~m).astype(np.float32)

            # En küçük iki uzaklığı güncelle(farklı nesnelere olan)
            d2 = np.where(d < d1, d1, np.minimum(d2, d))
            d1 = np.minimum(d1, d)

        fg_frac = foreground.mean()
        wc = np.where(
            foreground,
            0.5 / max(fg_frac, 1e-6),        # nesne pikseli
            0.5 / max(1 - fg_frac, 1e-6),    # arka plan pikseli
        )

        # Sınır terimi: birbirine yakın nesneler arasındaki arka plan şeridi
        if len(mask_files) >= 2:
            border = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
            border[foreground] = 0           # sadece arka plan pikselleri için
        else:
            border = np.zeros((H, W), dtype=np.float32)

        # Nihai ağırlık: w(x) = wc(x) + sınır terimi
        wmap = (wc + border).astype(np.float32)

        np.save(out_file, wmap)

    print("işlem tamamlandı.")

    return Path(out_dir)




# Kullanım


def train_validation_split(train_root : str,test_size : float = 0.2):
    root = Path(train_root)
    all_ids =sorted(p.name for p in root.iterdir() if p.is_dir())

    output_path = Path("splits.json").resolve()

    train_ids,val_ids = train_test_split(all_ids, test_size=test_size, random_state=42)

    with open(output_path, "w") as f:
        json.dump({"train": train_ids,
                   "validation": val_ids}, f)

    print("işlem tamamlandı")

    return output_path


