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
            assert len(self._images) == n, "Limiting dataset failed"

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
        img_name = self._image_name(idx)
        img = self._load_image(idx).to(device=self._device)
        mask = self._load_masks(idx).to(device=self._device)
        return img_name, img, mask

    def __len__(self):
        return len(self._images)

    def image_patch_index(self, loader_idx):
        """
        Return a tuple consisting of the image index and the patch index for 
        Return a tuple consisting of the image index and the patch index for 
        this respective loader index.
        """
        return loader_idx, 0

    def _image_name(self, loader_idx):
        image_idx, patch_idx = self.image_patch_index(loader_idx)
        img_path, _ = self._images[image_idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        return img_name

    def _load_image(self, loader_idx):
        image_idx, patch_idx = self.image_patch_index(loader_idx)
        img_path, _ = self._images[image_idx]
        img = Image.open(img_path)
        img = self.transform(img, patch_idx)
        return img

    def _load_masks(self, loader_idx):
        image_idx, patch_idx = self.image_patch_index(loader_idx)
        _, mask_paths = self._images[image_idx]
        masks = [None for _ in range(len(mask_paths))]
        for i, mask_path in enumerate(mask_paths):
            # this mask layer may be empty; only load images if non-empty
            # handle empty masks below
            if mask_path:
                mask = Image.open(mask_path)
                if mask.getbands() != ("P", ):
                    logger.warning(f"Processing mask with non-binary bands: {mask_path} has bands {mask.getbands()}")
                    mask = mask.convert("P")
                mask = self.mask_transform(mask, patch_idx)
                mask = mask.squeeze()
                mask[mask > 0] = 1
                masks[i] = mask
                logger.debug(f"Found mask for class {self.CLASSES[i]} (dims: {mask.size()}): {os.path.basename(mask_path)}")
            else:
                logger.debug(f"No mask found for class {self.CLASSES[i]}")

        # determine the size of the masks by any mask in the stack
        size_stack = torch.stack([torch.tensor(i.size()) for i in masks if i is not None], dim=1)
        masksize = tuple(torch.min(size_stack, dim=1).values)
        for i, mask in enumerate(masks):
            # fill in only empty masks
            if mask is not None:
                assert mask_paths[i] is not None, f"Inconsistent mask parsing"
                continue
            masks[i] = torch.zeros(size=masksize)

        for i, mask in enumerate(masks):
            pixels = torch.sum(mask[mask > 0])
            logger.debug("{s}, pixels = {pixels}, {p}".format(s=mask.shape, pixels=pixels, p=mask_paths[i]))

        # reshaping masks into format:
        # [classes, width, height]
        # one "layer", i.e. mask per class
        masks = torch.stack(masks, dim=-1)
        masks = masks.squeeze(0)  # remove empty first dimension
        masks = masks.permute(2, 0, 1)  # move mask dimension to front for compat with pytorch
        return masks

    def transform(self, image, patch_idx=None):
        """Take a PIL image, apply transforms and return a tensor."""
        t = transforms.Compose((
            transforms.Resize(256),
            transforms.ToTensor(),
        ))
        return t(image)

    def mask_transform(self, mask, patch_idx=None):
        return self.transform(mask, patch_idx)


class PatchIDRIDDataset(IDRIDDataset):
    # Transform single images into multiple patches
    def __init__(self, *args, patch_size=256, **kwargs):
        """
        Split every image from disk into `patch_size`**2 sized chunks and consider them independent samples.
        """
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
        # FIXME: limiting does not consider patches, but only images

    @property
    def _image_dims(self):
        return 4288, 2848

    @property
    def _patch_number(self):
        # FIXME: this depends on all images having the same size
        w, h = self._image_dims
        return (w // self._patch_size) * (h // self._patch_size)

    def image_patch_index(self, loader_idx):
        """
        Return a tuple consisting of the image index and the patch index for 
        this respective loader index.
        """
        image_idx = loader_idx // self._patch_number
        patch_idx = loader_idx % self._patch_number

        logger.debug(f"Loader index {loader_idx} --> image {image_idx}, patch {patch_idx}")
        return image_idx, patch_idx

    def __len__(self):
        return super().__len__() * self._patch_number

    def transform(self, image, patch_idx):
        # crop image first
        assert image.size == self._image_dims, f"Bad image size: {image.size}"
        w, h = self._image_dims
        # FIXME: biased against "end" of image - might never be included!
        patches_w = w // self._patch_size
        patches_h = h // self._patch_size

        logger.debug(f"{image.size} has room for {patches_w} x {patches_h} patches")
        w_idx = patch_idx % patches_w
        h_idx = patch_idx // (patches_h + 1)

        w0 = w_idx * self._patch_size
        w1 = w0 + self._patch_size
        h0 = h_idx * self._patch_size
        h1 = h0 + self._patch_size

        # (left, upper, right, lower)
        bbox = (w0, h0, w1, h1)
        logger.debug(f"Box for patch {patch_idx}: {bbox}")
        crop = image.crop(bbox)
        t = transforms.Compose((
            transforms.RandomCrop(224),  # increase data variance
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ))
        return t(crop)


class BinaryPatchIDRIDDataset(PatchIDRIDDataset):
    def __init__(self, *args, presence_threshold=100, **kwargs):
        """
        Dataset, that splits images into even patches (`patch_size`**2 in size) and calculates
        presence and absence of features (loaded from mask files), giving a binary value for every
        marker.
        """
        super().__init__(*args, **kwargs)
        self._presence_threshold = presence_threshold

    def __getitem__(self, idx):
        img_name, img, masks = super().__getitem__(idx)
        # TODO: There is certainly a faster way to do this that is still readable
        sum_masks = [torch.sum(mask[mask > 0]) for mask in masks]
        # without using ().value, this would be a list of 1-element 1-d tensors
        boolean_mask = [(mask > self._presence_threshold).item() for mask in sum_masks]
        binary_mask = [{False: 0, True: 1}[m] for m in boolean_mask]
        return img_name, img, torch.tensor(binary_mask, dtype=torch.float)


if __name__ == "__main__":
    print(f"Exploring test dataset at {DEFAULT_DATA_DIR}")

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('sample', help="Display this sample", type=int)
    p.add_argument('-p', '--patched', default=-1, type=int, help="Use patched dataloader with this size")
    args = p.parse_args()
    print(f"Displaying sample {args.sample}")
    if args.patched > 0:
        ds = PatchIDRIDDataset("test", patch_size=args.patched)
    else:
        ds = IDRIDDataset("test")

    assert args.sample <= len(ds), f"Sample {args.sample} does not exist; largest sample is {len(ds)}"

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    dname, di, dm = ds[args.sample]  # get an image and its list of masks from the dataset
    print(dname)
    di = di.permute(1, 2, 0)  # reorder dims

    nrows = math.floor(math.sqrt(len(ds.CLASSES) + 1))
    ncols = nrows + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    axes = axes.flatten()  # don't care about the order of the axes

    axes[0].set_title(dname)
    axes[0].imshow(di)

    for i, cls in enumerate(ds.CLASSES):
        mask = dm[i, :, :]
        pixels = torch.sum(mask[mask > 0])
        axes[i + 1].imshow(di, alpha=1.0)
        axes[i + 1].imshow(mask, alpha=0.5, cmap=cm.binary)
        axes[i + 1].set_title(f"{cls}; px={pixels}")

    plt.show()
