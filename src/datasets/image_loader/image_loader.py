from os import listdir
from os.path import join, isfile, splitext, basename
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.transforms import functional as F
from torchvision.io.image import ImageReadMode
from utils.helper_funcs import (
    calc_edge,
    calc_distance_map,
    normalize
)

np_normalize = lambda x: (x - x.min()) / (x.max() - x.min())


def calculate_image(image_size, origin_image_size):
    aspect_ratio = origin_image_size[1] / origin_image_size[0]
    return aspect_ratio, (int(image_size * aspect_ratio) - int(image_size * aspect_ratio) % 16, image_size)


class ImageLoader(Dataset):
    def __init__(
            self,
            mode,
            data_dir=None,
            one_hot=True,
            image_size=224,
            aug=None,
            aug_empty=None,
            transform=None,
            img_transform=None,
            msk_transform=None,
            add_boundary_mask=False,
            add_boundary_dist=False,
            support_types: [str] = None
    ):
        """Initializes image paths and preprocessing module."""
        if support_types is None:
            support_types = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF']
        support_types = set(support_types)
        # pre-set variables
        self.data_dir = data_dir if data_dir else "/path/to/datasets/ISIC2018"

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.aug = aug
        self.aug_empty = aug_empty
        self.transform = transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode

        self.add_boundary_mask = add_boundary_mask
        self.add_boundary_dist = add_boundary_dist

        self.origin_image_path = join(self.data_dir, "images")
        self.ground_truth_path = join(self.data_dir, "masks")
        self.image_paths = [join(self.origin_image_path, f) for f in listdir(self.origin_image_path)
                            if isfile(join(self.origin_image_path, f)) and splitext(f)[1][1:] in support_types]
        self.image_size = image_size
        self.mode = mode
        print(f"Dataset Type: {self.mode}, image count: {len(self.image_paths)}")

    def get_gt_file_name(self, origin_image_name: str, extension: str) -> str:
        assert False, "Need to implement this method in child class"
        return ""

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        origin_image_name, extension = splitext(basename(image_path))
        filename = self.get_gt_file_name(origin_image_name, extension)
        GT_path = join(self.ground_truth_path, filename)
        img = read_image(image_path, ImageReadMode.RGB)
        msk = read_image(GT_path, ImageReadMode.GRAY)

        if self.one_hot:
            msk = (msk - msk.min()) / (msk.max() - msk.min())
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        if self.aug:
            if self.mode == "tr":
                img_ = np.uint8(torch.moveaxis(img * 255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk * 255, 0, -1).detach().numpy())
                augmented = self.aug(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
            elif self.aug_empty:  # "tr", "vl", "te"
                img_ = np.uint8(torch.moveaxis(img * 255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk * 255, 0, -1).detach().numpy())
                augmented = self.aug_empty(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
            img = img.nan_to_num(127)
            img = normalize(img)
            msk = msk.nan_to_num(0)
            msk = normalize(msk)

        if self.add_boundary_mask or self.add_boundary_dist:
            msk_ = np.uint8(torch.moveaxis(msk * 255, 0, -1).detach().numpy())

        if self.add_boundary_mask:
            boundary_mask = calc_edge(msk_, mode='canny')
            # boundary_mask = np_normalize(boundary_mask)
            msk = torch.concatenate([msk, torch.tensor(boundary_mask).unsqueeze(0)], dim=0)

        if self.add_boundary_dist:
            boundary_mask = boundary_mask if self.add_boundary_mask else calc_edge(msk_, mode='canny')
            distance_map = calc_distance_map(boundary_mask, mode='l2')
            distance_map = distance_map / (self.image_size * 1.4142)
            distance_map = np.clip(distance_map, a_min=0, a_max=0.2)
            distance_map = (1 - np_normalize(distance_map)) * 255
            msk = torch.concatenate([msk, torch.tensor(distance_map).unsqueeze(0)], dim=0)

        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform:
            msk = self.msk_transform(msk)

        img = img.nan_to_num(0.5)
        msk = msk.nan_to_num(-1)

        sample = {"image": img, "mask": msk, "id": index, "origin_image_name": origin_image_name}
        return sample

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)
