import json

import cv2
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from torchvision import tv_tensors
import albumentations as A

train_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent=(-0.1, 0.1),
        rotate=(-180, 180),
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.8,
    ),
    A.ElasticTransform(
        alpha=50, sigma=7,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.5,
    ),
    # sadece image'a uygulanır, mask'lere dokunmaz
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
])

from typing import Literal

class NucleiDataset(Dataset):
    def __init__(self, root, weight_dir=None, split_file=None, split=None,
                 augment=False, tile=True, has_masks=True,
                 in_size=572, out_size=388):
        self.root = Path(root)
        self.weight_dir = Path(weight_dir) if weight_dir is not None else None
        self.augment = augment
        self.tile = tile                  # train/val: True, test: False
        self.has_masks = has_masks
        self.in_size, self.out_size = in_size, out_size

        if split_file is not None:
            if split is None:
                raise ValueError("split_file verildiyse split de verilmeli")
            with open(split_file) as f:
                self.ids = json.load(f)[split]
        else:
            self.ids = sorted(p.name for p in self.root.iterdir() if p.is_dir())

        if (augment or tile) and (not has_masks or self.weight_dir is None):
            raise ValueError("tile/augment için maske ve ağırlık haritası gerekli")

    def __len__(self):
        return len(self.ids)

    def _load_mask(self, sample_id):
        mask_files = sorted((self.root / sample_id / "masks").glob("*.png"))
        first = np.array(Image.open(mask_files[0]).convert("L"))
        foreground = np.zeros(first.shape, dtype=np.uint8)
        for f in mask_files:
            foreground |= (np.array(Image.open(f).convert("L")) > 127).astype(np.uint8)
        return foreground

    def _load(self, sample_id):
        img_path = next((self.root / sample_id / "images").glob("*.png"))
        img = np.array(Image.open(img_path).convert("RGB"))

        mask = self._load_mask(sample_id) if self.has_masks else None
        wmap = (np.load(self.weight_dir / f"{sample_id}.npy")
                if self.weight_dir is not None else None)
        return img, mask, wmap

    def __getitem__(self, index):
        sample_id = self.ids[index]
        img, mask, wmap = self._load(sample_id)

        if self.augment:
            out = train_augmentation(image=img, masks=[mask, wmap])
            img, (mask, wmap) = out["image"], out["masks"]

        if self.tile:   # train ve validation
            img_t, mask_t, w_t = extract_tile(img, mask, wmap, random_pos=True)
            return (torch.from_numpy(img_t).float().permute(2, 0, 1) / 255.0,
                    torch.from_numpy(mask_t).long(),
                    torch.from_numpy(w_t).float())

        # test: tam görüntü, tile'lama tahmin aşamasında yapılır
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask).long() if mask is not None else torch.empty(0)
        return img_t, mask_t, sample_id


train_root = Path(r"C:\Users\omen\PycharmProjects\UNetArchitecture\stage1_train")
weight_dir = Path(r"C:\Users\omen\PycharmProjects\UNetArchitecture\stage1_weightmaps")


def create_data_loaders(train_path,weight_dir,test_path,split_file):
    train_ds = NucleiDataset(root = train_path,weight_dir=weight_dir,split_file=split_file,
                             split="train",augment=True,tile=True,has_masks=True)

    val_ds = NucleiDataset(root = train_path,weight_dir=weight_dir,split_file=split_file,
                             split="validation",augment=True,tile=True,has_masks=True)

    test_ds = NucleiDataset(test_path,augment=False,tile=False,has_masks=False,in_size=572, out_size=388)

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_dataloader,val_dataloader,test_dataloader





















import random

IN_SIZE, OUT_SIZE = 572, 388
MARGIN = (IN_SIZE - OUT_SIZE) // 2   # 92

def extract_tile(image, mask, weight, random_pos=True):

    H,W = mask.shape

    ph,pw = max(0,OUT_SIZE-H),max(0,OUT_SIZE-W)

    if ph or pw:
        image = np.pad(image,((0,ph),(0,pw),(0,0)),"reflect")
        mask = np.pad(mask,((0,ph),(0,pw)),"reflect")
        weight = np.pad(weight,((0,ph),(0,pw)),"reflect")

        H,W = mask.shape


    if random_pos:
        y = random.randint(0,H-OUT_SIZE)
        x = random.randint(0,W-OUT_SIZE)
    else:
        y = (H - OUT_SIZE) // 2
        x = (W - OUT_SIZE) // 2


    image_pad = np.pad(image,((MARGIN,MARGIN),(MARGIN,MARGIN),(0,0)),"reflect")

    tile_image = image_pad[y : y + IN_SIZE , x : x + IN_SIZE]
    tile_mask = mask[y : y + OUT_SIZE , x : x + OUT_SIZE]
    tile_weight = weight[y : y + OUT_SIZE , x : x + OUT_SIZE]

    return tile_image,tile_mask,tile_weight


