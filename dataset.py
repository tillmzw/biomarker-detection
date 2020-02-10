#!/usr/bin/env python3

import os
import logging
import math
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils import class_weight


logger = logging.getLogger(__name__)


class RetinopathyDataset(Dataset):
    def __init__(self, labels_file, path, limit=None, device="cpu"):
        assert os.path.isfile(labels_file), "No such file: %s" % labels_file
        assert os.path.isdir(path), "No such directory: %s" % path

        self._device = torch.device(device)

        start = time.time()
        self._labels = pd.read_csv(labels_file, index_col=0)
        self._labels["patient"] = self._labels.index.map(lambda x: x.split('_')[0])
        self._labels["eye"] = self._labels.index.map(lambda x: 1 if x.split('_')[-1] == 'left' else 0)
        self._labels["path"] = self._labels.index.map(lambda x: os.path.join(os.path.abspath(path), "%s.jpeg" % x))
        self._labels["file_exists"] = self._labels["path"].map(os.path.exists)
        # limit to images found on disk
        self._labels = self._labels[self._labels['file_exists']]

        if limit:
            if limit < 1:
                limit_n = math.floor(len(self._labels.index) * limit)
            else:
                limit_n = math.floor(limit)
            logger.warning("Limiting dataset to %d images (of %d)" % (limit_n, len(self._labels.index)))
            self._labels = self._labels.sample(limit_n)

        logger.info("Loading %d images from %s took %.2f s" % (len(self), path, (time.time() - start)))

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        """Retrieve the image at `idx` and the corresponding label"""
        label_row = self._labels.iloc[idx]
        image = label_row["path"]
        image_repr = "%s_%s" % (label_row["patient"], label_row["eye"])
        label = label_row.level
        return self.load(image), image_repr, label

    def load(self, image):
        logger.debug("Loading image %s from path %s" % (os.path.basename(image), os.path.dirname(image)))
        start = time.time()
        img = Image.open(image)
        timg = self.transform(img)
        logger.debug("Loading image took %.3f seconds" % (time.time() - start))
        return timg

    def transform(self, image):
        start = time.time()
        # see https://pytorch.org/hub/pytorch_vision_wide_resnet/
        transform = transforms.Compose([
            transforms.Resize(256),
            #transforms.CenterCrop(224),
            # channel shifts -- brightness, contrasts, ...
            transforms.RandomCrop(224),  # increase data variance
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)
        logger.debug("Preprocessing image took %.3f seconds" % (time.time() - start))
        start = time.time()
        image = image.to(self._device)
        logger.debug("Moving image to device %s took %.3f seconds" % (self._device, time.time() - start))

        return image

    def classes(self, unique=True):
        labels = np.squeeze(self._labels["level"].to_numpy())
        if unique:
            labels = np.unique(labels)
        return labels

    def class_weights(self):
        """Calculate the weight for each class in this dataset. Returns a 1D tensor."""
        logger.info("Calculating class weights")

        weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=self.classes(unique=True),
                y=self.classes(unique=False))

        if np.squeeze(weights.shape) == 1:
            logger.warning("Skipping weights, only detected a single class in the data!")
            return None
        if np.squeeze(weights.shape) != 5:
            raise RuntimeError("Need exactly 5 classes in the sample, found %d" % np.squeeze(weights.shape))
        return torch.from_numpy(weights).float().to(self._device)
