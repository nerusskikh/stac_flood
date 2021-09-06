import rasterio
import numpy as np
import albumentations as A
import cv2
import torch
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

from config import *


def read_img_and_mask(fp):
    with rasterio.open(fp) as f:
        masked_arr = f.read(1, masked=True)
        data = masked_arr.data
        mask = masked_arr.mask
    return data, mask

 
def img2tensor(img, dtype=np.float32):
    if img.ndim == 2 : img = np.expand_dims(img, -1)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def get_aug(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1,
                            rotate_limit=15,
                            p=0.9, 
                            border_mode=cv2.BORDER_REFLECT),
    ], p=p)


class STACDataset(Dataset):
    def __init__(self, fold, train=True, tfms=None):
        self.nfolds = nfolds
        kf = GroupKFold(n_splits=nfolds)
        ids = [fname[:5] for fname in os.listdir(LABELS)]
        groups = [fname[:3] for fname in os.listdir(LABELS)]
        self.fold_ids = [ids[i] for i in list(kf.split(ids, groups=groups))[fold][0 if train else 1]]
        self.train = train
        self.tfms = tfms
        
    def __len__(self):
        return len(self.fold_ids)
    
    def __getitem__(self, idx):
        sample_id = self.fold_ids[idx]
        
        fp_vv = os.path.join(TRAIN,sample_id+'_vv.tif')
        img_vv, mask_vv = read_img_and_mask(fp_vv)
        
        fp_vh = os.path.join(TRAIN,sample_id+'_vh.tif')
        img_vh, mask_vh = read_img_and_mask(fp_vh)
        
        fp_target = os.path.join(LABELS, sample_id+'.tif')
        target, mask_target = read_img_and_mask(fp_target)
        
        img = np.stack([img_vv, img_vh], axis=-1)
        
        # Min-max normalization
        min_norm = -77
        max_norm = 26
        img = np.clip(img, min_norm, max_norm)
        img = (img - min_norm) / (max_norm - min_norm)

        
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=target)
            img, target = augmented['image'], augmented['mask']
        return img2tensor(img), img2tensor(target, dtype=np.int64)
