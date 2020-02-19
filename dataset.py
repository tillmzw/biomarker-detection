#!/usr/bin/env python3

import os
import logging
import math
import time
import glob
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils import class_weight


logger = logging.getLogger(__name__)
PARENT_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PARENT_DIR, "data")


class IDRIDDataset(Dataset):

    CLASSES = ("Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc")

    def __init__(self, itype, path=DEFAULT_DATA_DIR, limit=None, device="cpu"):
        self._device = torch.device(device)
        self._images = self._extract_image_paths(itype, path)
        if limit is not None:
            if limit < 1:
                # limit is a percentage
                n = math.floor(len(self._images) * limit)
            else:
                n = int(limit)
            logger.info(f"Limiting number of images to {n} (based on limit = {limit})")
            self._images = random.sample(self._images, n)

    def _extract_image_paths(self, itype, path):
        if itype == "train":
            itype_dir = "a. Training Set"
        elif itype == "test":
            itype_dir = "b. Testing Set"
        else:
            raise RuntimeError(f"Don't know how to find image type: {itype}")

        image_dir = os.path.join(path, "1. Original Images", itype_dir)
        truth_dir = os.path.join(path, "2. All Segmentation Groundtruths", itype_dir)
        assert os.path.isdir(image_dir), f"Bad folder structure in {path}: {image_dir} doesn't exist"
        assert os.path.isdir(truth_dir), f"Bad folder structure in {path}: {truth_dir} doesn't exist"

        class_dir_names = [f"{prefix}. {cls}" for prefix, cls in zip(range(1, 6), self.CLASSES)]
        class_dir_paths = [os.path.join(truth_dir, cls) for cls in class_dir_names]
        for class_dir_path in class_dir_paths:
            assert os.path.isdir(class_dir_path), f"Directory {class_dir_path} absent"

        """ Build the following structure:
        (
            ("p1/p2/image_001.jpg", ("t1/c1/image_001.tif", None, "t1/tc2/image_001.tif", ...))
        )
        """
        images = [
            (os.path.join(image_dir, image), self._mask_paths_for_image(image, class_dir_paths)) 
            for image in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, image))
        ]

        return images

    def _mask_paths_for_image(self, img, class_dir_paths):
        """
        Return a list of paths, for every element in `CLASSES`.
        """
        masks = [None for _ in class_dir_paths]
        # remove extension
        img_name, _ = os.path.splitext(img)

        for i, class_dir_path in enumerate(class_dir_paths):
            # FIXME: path separator
            matches = glob.glob(f"{class_dir_path}/{img_name}*")
            assert len(matches) < 2, f"Found more than one mask for {img_name} and type {class_dir_path}"
            mask_path = matches[0] if len(matches) == 1 else None
            masks[i] = mask_path
        return masks

    def __getitem__(self, idx):
        img_path, mask_paths = self._images[idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        img = self._load_image(img_path)
        mask = self._load_masks(mask_paths)
        return img_name, img, mask

    def __len__(self):
        return len(self._images)

    def _load_image(self, img_path):
        img = Image.open(img_path)
        img = self.normalize_transform(img)
        # TODO: further transformations
        return img

    def _load_masks(self, mask_paths):
        imgs = [None for _ in range(len(mask_paths))]
        for i, mask_path in enumerate(mask_paths):
            # this mask layer may be empty; only load images if non-empty
            # handle empty masks below
            if mask_path:
                img = Image.open(mask_path)
                if img.getbands() != ("P", ):
                    logger.warning(f"Processing mask with non-binary bands: {mask_path} has bands {img.getbands()}")
                    img = img.convert("P")
                img = self.normalize_transform(img)
                img = img.squeeze()
                img[img > 0] = 1
                imgs[i] = img
                logger.debug(f"Found mask for class {self.CLASSES[i]} (dims: {img.size()}): {os.path.basename(mask_path)}")
            else:
                logger.debug(f"No mask found for class {self.CLASSES[i]}")

        # determine the size of the masks by any mask in the stack
        size_stack = torch.stack([torch.tensor(i.size()) for i in imgs if i is not None], dim=1)
        masksize = tuple(torch.min(size_stack, dim=1).values)
        for i, img in enumerate(imgs):
            # fill in only empty masks
            if img is not None:
                assert mask_paths[i] is not None, f"Inconsistent mask parsing" 
                continue
            imgs[i] = torch.zeros(size=masksize)

        for i, img in enumerate(imgs):
            pixels = torch.sum(img[img > 0])
            logger.debug("{s}, pixels = {pixels}, {p}".format(s=img.shape, pixels=pixels, p=mask_paths[i]))

        # reshaping masks into format:
        # [samples, classes, width, height]
        # one "layer", i.e. mask per class
        masks = torch.stack(imgs, dim=-1)
        masks = masks.squeeze(0)  # remove empty first dimension
        masks = masks.permute(2, 0, 1)  # move mask dimension to front for compat with pytorch
        return masks

    def normalize_transform(self, image):
        """Apply transforms such that images are in normalized size format."""
        t = transforms.Compose((
            transforms.Resize(256),
            transforms.ToTensor(),
        ))
        image = t(image).to(self._device)
        return image

    def transform(self, image):
        # see https://pytorch.org/hub/pytorch_vision_wide_resnet/
        transform = transforms.Compose([
            transforms.RandomCrop(224),  # increase data variance
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).to(self._device)
        return image


class PatchIDRIDDataset(IDRIDDataset):
    # TODO: Transform a single images into multiple patches
    pass


if __name__ == "__main__":
    print(f"Exploring test dataset at {DEFAULT_DATA_DIR}")

    # TODO: argparser to show a specific image
    ds = IDRIDDataset("test")

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    dname, di, dm = ds[0]  # get an image and its list of masks from the dataset
    di = di.permute(1, 2, 0)  # reorder dims

    nrows = math.floor(math.sqrt(len(ds.CLASSES) + 1))
    ncols = nrows + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    axes = axes.flatten()  # don't care about the order of the axes

    axes[0].set_title(dname)
    axes[0].imshow(di)

    for i, cls in enumerate(ds.CLASSES):
        print(dm.shape)
        mask = dm[i, :, :]
        axes[i + 1].imshow(di, alpha=1.0)
        axes[i + 1].imshow(mask, alpha=0.5, cmap=cm.binary)
        axes[i + 1].set_title(cls)

    plt.show()
