#!/usr/bin/env python3

import os
import logging
import math
import datetime
import glob
import random
from typing import Tuple
import utils

import torch
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils import class_weight


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
PARENT_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PARENT_DIR, "data")


class IDRIDDataset(Dataset):

    CLASSES = ("Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc")

    def __init__(self, itype, path=DEFAULT_DATA_DIR, limit=None, device="cpu"):
        # TODO: in some cases we want to move tensors to CUDA, but in most cases we must retain them on the CPU
        #        -  organize!
        self._itype = itype
        self._path = path
        self._device = torch.device(device)
        self._images = self._extract_image_paths()
        n_images_unlimited = len(self._images)
        if limit is not None:
            if limit < 1:
                # limit is a percentage and should result in at least one image
                n = max((math.floor(len(self._images) * limit), 1))
            else:
                n = int(limit)
            self._limit = n
            self._limit_dataset()
        else:
            self._limit = None

        logger.info(f"Using {len(self)} samples from {n_images_unlimited} available images for type {self._itype}")

    def _limit_dataset(self):
        # Move this to a method to allow children to override this.
        logger.debug(f"Limiting number of images in {self._itype} to {self._limit}")
        self._images = random.sample(self._images, self._limit)

    def _extract_image_paths(self):
        if self._itype == "train":
            itype_dir = "a. Training Set"
        elif self._itype == "test":
            itype_dir = "b. Testing Set"
        else:
            raise RuntimeError(f"Don't know how to find image type: {self._itype}")

        image_dir = os.path.join(self._path, "1. Original Images", itype_dir)
        truth_dir = os.path.join(self._path, "2. All Segmentation Groundtruths", itype_dir)
        assert os.path.isdir(image_dir), f"Bad folder structure in {self._path}: {image_dir} doesn't exist"
        assert os.path.isdir(truth_dir), f"Bad folder structure in {self._path}: {truth_dir} doesn't exist"

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
        return img_name, self.transform(img), self.mask_transform(mask)

    def __len__(self):
        return len(self._images)

    def image_patch_index(self, loader_idx):
        """
        Return a tuple consisting of the image index and the patch index for 
        this respective loader index.
        """
        return loader_idx, 0

    def get_path_for_name(self, image_name):
        for image_idx, (img_path, _) in enumerate(self._images):
            img_name, _ = os.path.splitext(os.path.basename(img_path))
            if img_name == image_name:
                return os.path.abspath(img_path)

    def _image_name(self, image_idx):
        img_path, _ = self._images[image_idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))
        return img_name

    def _load_image(self, image_idx) -> torch.Tensor:
        img_path, _ = self._images[image_idx]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img).to(device=self._device, non_blocking=True)
        return img

    def _load_masks(self, image_idx) -> torch.Tensor:
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
                mask = transforms.ToTensor()(mask).to(device=self._device)  # can be blocking, since we work on them any way
                mask = mask.squeeze()
                mask[mask > 0] = 1
                masks[i] = mask
                logger.debug(f"Found mask for class {self.CLASSES[i]} (dims: {mask.size()}): {os.path.basename(mask_path)}")
            else:
                logger.debug(f"No mask found for class {self.CLASSES[i]}")

        # determine the size of the masks by any mask in the stack
        size_stack = torch.stack([torch.tensor(i.size()) for i in masks if i is not None], dim=1)
        masksize = tuple(torch.min(size_stack, dim=1).values)
        assert masksize == self._image_dims[::-1], f"Mask size {masksize} not matching image size {self._image_dims}"
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

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def mask_transform(self, mask: torch.Tensor) -> torch.Tensor:
        return mask

    @property
    def _image_dims(self):
        return 4288, 2848

    @property
    def _sample_dims(self):
        return self._image_dims

    @property
    def _mask_dims(self):
        return self._sample_dims


class CachedIDRIDDataset(IDRIDDataset):

    GROUP_IMAGES = "images"
    GROUP_MASKS = "masks"

    def __init__(self, *args, cache_file_name="cache.hdf5", **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(f"cache.{self._itype}")
        self.logger.setLevel(logging.INFO)
        self._cache_file_name = os.path.join(self._path, cache_file_name)
        self._cache_fh = None
        self._cache = None

    def __del__(self):
        if self._cache_fh:
            try:
                self._cache_fh.close()
            except ImportError as ie:
                self.logger.error(f"An error occured when closing the cache: {ie}")
            self.logger.info(f"Closed cache at {self._cache_file_name}")

    def _load_image(self, image_idx) -> torch.Tensor:
        self.logger.debug(f"Retrieving image {image_idx}")
        img = self._cache_retrieve(self.GROUP_IMAGES, image_idx)
        # TODO: is this a valid way to determine whether this attribute has been set?
        if img is not None and img.max() > 0:
            return torch.tensor(img).to(device=self._device, non_blocking=True, dtype=torch.float32)
        else:
            return super()._load_image(image_idx)

    def _load_masks(self, image_idx) -> torch.Tensor:
        self.logger.debug(f"Retrieving masks {image_idx}")
        mask = self._cache_retrieve(self.GROUP_MASKS, image_idx)
        # TODO: is this a valid way to determine whether this attribute has been set?
        if mask is not None and mask.max() > 0:
            return torch.tensor(mask).to(device=self._device, non_blocking=True, dtype=torch.float32)
        else:
            return super()._load_masks(image_idx)

    def _require_cache(self):
        if self._cache_fh is None:
            self._cache_fh = h5py.File(self._cache_file_name, "r")

        if self._itype not in self._cache_fh.keys():
            raise RuntimeError(f"Cache not initialized")

        self._cache = self._cache_fh[self._itype]

        for dataset in (self.GROUP_IMAGES, self.GROUP_MASKS):
            if dataset not in self._cache.keys():
                raise RuntimeError(f"Dataset {dataset} is not initialized.")

        self.logger.info(f"Loaded cache at {self._cache_file_name}")

    def _cache_retrieve(self, group, idx):
        if self._cache is None:
            self._require_cache()

        self.logger.debug(f"Retrieving item from group {group} for index {idx}")
        try:
            return self._cache[group][idx]
        except KeyError:
            self.logger.warning(f"No cache hit at {group} for index {idx}")
            return None

    def initialize(self, force=False):
        self.logger.info(f"Starting initialization of cache at {self._cache_file_name} with {self.__len__()} items")
        assert self._cache_fh is None, f"Cache file handle must be unused when initializing"
        """
        # TODO: Create a user block in the HDF5 file - impossible to do with this library, and manually doing it
                corrupts the file signature (probably because the user block is not at the very beginning of the file)
        USERBLOCK_SIZE = 512
        if not os.path.isfile(self._cache_file_name):
            # create the file with a user block first.
            self.logger.info("Creating new HDF5 file with user block")
            with open(self._cache_file_name, "w") as f:
                f.seek(0)
                text = f"Created by version: {utils.git_hash()}\n Date: {datetime.datetime.now()}\n"
                contents = text.ljust(USERBLOCK_SIZE, "\0")
                assert len(contents) == USERBLOCK_SIZE, "Invalid user block size for HDF5 file"
                f.writelines(contents)
        """
        self._cache_fh = h5py.File(self._cache_file_name, "a", userblock_size=0)

        self.logger.debug(f"Creating root group {self._itype}")
        self._cache = self._cache_fh.require_group(self._itype)

        # TODO: dynamically calculate the shape, or dont duplicate child classes here
        img_shape = (self.__len__(), 3) + self._sample_dims[::-1]  # reverse order of dimensions here
        mask_shape = (self.__len__(), 5) + self._mask_dims[::-1]  # reverse order of dimensions here

        self.logger.debug(f"Creating dataset {self.GROUP_IMAGES} with shape {img_shape}")
        self._cache.require_dataset(self.GROUP_IMAGES, shape=img_shape, dtype=np.float32)

        self.logger.debug(f"Creating dataset {self.GROUP_MASKS} with shape {mask_shape}")
        self._cache.require_dataset(self.GROUP_MASKS, shape=mask_shape, dtype=np.float32)

        self.logger.debug(f"Group {self._itype} has datasets: {self._cache.keys()}")

        # Actually start to store the data
        ds_i = self._cache[self.GROUP_IMAGES]
        ds_m = self._cache[self.GROUP_MASKS]
        for i, data in enumerate(tqdm(self, unit="sample")):
            # TODO: `i` will increase, even though the randomized dataset will yield the same image many times
            #       Current solution: Don't run this via any randomized dataset.
            img, mask = data[-2:]
            # TODO: is this a valid way to determine whether this attribute has been set?
            if force or ds_i[i].max() == 0:
                self.logger.debug(f"Storing image at index {i}")
                ds_i[i] = img
            else:
                self.logger.debug(f"Image at {i} already exists; skipping")
            if force or ds_m[i].max() == 0:
                self.logger.debug(f"Storing mask at index {i}")
                ds_m[i] = mask
            else:
                self.logger.debug(f"Mask at {i} already exists; skipping")

        self.logger.info("Initialization finished. Closing file.")

        self._cache_fh.close()


class RandomPatchIDRIDDataset(CachedIDRIDDataset):
    def __init__(self, *args, patch_size=256, n_patches=100, **kwargs):
        """
        Split every image from disk into `n_patches` times `patch_size`**2-sized chunks and consider them independent samples.
        Note: These chunks may overlap!
        """
        self._patch_size = patch_size
        self._n_patches = n_patches
        super().__init__(*args, **kwargs)

    def _limit_dataset(self):
        # prefer to take one sample per image instead of 10 samples per image if limit is 10
        # if limit > len, we limit the dataset in __len__
        if self._limit < len(self._images):
            self._images = random.sample(self._images, self._limit)

    def __len__(self):
        return min(self._limit or np.inf, self._n_patches * len(self._images))

    def __getitem__(self, idx) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int]], torch.Tensor, torch.Tensor]:
        image_idx = idx // self._n_patches

        img_path, _ = self._images[image_idx]
        img_name, _ = os.path.splitext(os.path.basename(img_path))

        # TODO: caching will never cover randomization - only the base images! Check whether this needs to changed
        image = self._load_image(image_idx)
        masks = self._load_masks(image_idx)

        tower = torch.cat((image, masks), dim=0)
        w, h = self._image_dims
        row_start = random.randint(0, (h - self._patch_size))
        row_end = row_start + self._patch_size

        col_start = random.randint(0, (w - self._patch_size))
        col_end = col_start + self._patch_size

        tower_cropped = tower[:, row_start:row_end, col_start:col_end]
        image_c = tower_cropped[0:3, :, :]
        masks_c = tower_cropped[3:8, :, :]

        assert image_c[0, :, :].shape == torch.Size((self._patch_size, self._patch_size)), "Bad cropping"

        image_ct = self.transform(image_c)
        masks_ct = self.mask_transform(masks_c)

        return img_name, ((row_start, col_start), (row_end, col_end)), image_ct, masks_ct

    @property
    def _sample_dims(self):
        return self._patch_size, self._patch_size


class BinaryPatchIDRIDDataset(RandomPatchIDRIDDataset):
    def __init__(self, *args, presence_threshold=100, **kwargs):
        """
        Dataset that splits images into even patches (`patch_size`**2 in size) and calculates
        presence and absence of features (loaded from mask files), giving a binary value for every
        marker. As such, the image itself can be exposed to random transforms without affecting the
        mask outputs.
        """
        super().__init__(*args, **kwargs)
        self._presence_threshold = presence_threshold

    @property
    def _mask_dims(self):
        return 5,

    def class_weights(self):
        if self._presence_threshold == 10 and self._patch_size == 500 and self._n_patches >= 50:
            # these values are empiric values from `./tools/label-count.py`
            # NOTE: this list is inverted, i.e. rare labels have higher weight
            return torch.tensor((0.470, 0.771, 0.720, 5.400, 3.600))
        else:
            logger.warning(f"No empiric class weights available for configuration "
                           f" -p [patch size] {self._patch_size},"
                           f" -P [presence threshold] {self._presence_threshold},"
                           f" -n [patch number] {self._n_patches}")
            return torch.tensor((1, 1, 1, 1, 1))

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        t = transforms.Compose((
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ))
        return t(image)

    def mask_transform(self, mask: torch.Tensor) -> torch.Tensor:
        sum_masks = [torch.sum(m[m > 0]) for m in mask]
        # without using ().value, this would be a list of 1-element 1-d tensors
        boolean_mask = [(mask > self._presence_threshold).item() for mask in sum_masks]
        binary_mask = [{False: 0, True: 1}[m] for m in boolean_mask]
        return torch.tensor(binary_mask, dtype=torch.float)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import utils

    print(f"Exploring test dataset at {DEFAULT_DATA_DIR}")

    p = argparse.ArgumentParser()
    p.add_argument('sample', nargs="?", default=None, help="Display this sample", type=int)
    p.add_argument('-p', '--patch-size', default=500, type=int, help="Use patches of this size")
    p.add_argument('-r', '--random', nargs="?", default=None, const=100, type=int, help="Use random patch dataloader with this many patches")
    args = p.parse_args()

    if args.random:
        ds = RandomPatchIDRIDDataset("test", patch_size=args.patch_size, n_patches=args.random)
    else:
        ds = IDRIDDataset("test")

    print("Using dataset %s" % ds.__class__.__name__)

    if args.sample is None:
        args.sample = random.randint(0, len(ds))
    print(f"Displaying sample {args.sample}")

    dname, coords, di, dm = ds[args.sample]  # get an image and its list of masks from the dataset
    print(dname)
    print("Sample taken from coordinates %s" % [",".join(str(c) for c in coords)])
    di = di.permute(1, 2, 0)  # reorder dims

    nrows = math.floor(math.sqrt(len(ds.CLASSES) + 1))
    ncols = nrows + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))
    fig.tight_layout()
    axes = axes.flatten()  # don't care about the order of the axes

    axes[0].set_title(dname)
    axes[0].imshow(di)
    # change tick labels so that the coordinates are included
    # need a drawn canvas so that axes ticks are computed
    fig.canvas.draw()
    # note: the ticks have funny minus sign chars...
    xlabels_off = [float(l.get_text().replace("−", "-")) + coords[0][1] for l in axes[0].get_xticklabels()]
    ylabels_off = [float(l.get_text().replace("−", "-")) + coords[0][0] for l in axes[0].get_yticklabels()]

    axes[0].set_xticklabels(xlabels_off)
    axes[0].set_yticklabels(ylabels_off)

    for i, cls in enumerate(ds.CLASSES):
        mask = dm[i, :, :]
        pixels = torch.sum(mask[mask > 0])
        # TODO: fix plotting -- rescale_vector is not (yet) compatible with >1d vectors
        di_s = utils.rescale_pixel_values(di)
        axes[i + 1].imshow(di_s, alpha=1.0)
        # TODO: fix cmap - cmap range might be wrong?
        axes[i + 1].imshow(mask.to(dtype=torch.float), alpha=0.5, cmap=cm.binary)
        axes[i + 1].set_title(f"{cls}; px={pixels}")

        fig.canvas.draw()
        xlabels_off = [float(l.get_text().replace("−", "-")) + coords[0][1] for l in axes[i + 1].get_xticklabels()]
        ylabels_off = [float(l.get_text().replace("−", "-")) + coords[0][0] for l in axes[i + 1].get_yticklabels()]

        axes[i+1].set_xticklabels(xlabels_off)
        axes[i+1].set_yticklabels(ylabels_off)

    plt.show()
