#!/usr/bin/env python3

import os
import csv
import logging
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix


logger = logging.getLogger(__name__)


def validate(net, dataloader, record_file=None):
    """Count the number of correct and incorrect predictions made by `net` on `dataloader`.
    Returns a percentage accuracy, Cohen's Kappa and a confusion matrix.
    """
    logger.info("Starting validation against test set")

    # set the network to eval mode
    net.eval()

    # FIXME: this only works for one GPU!
    model_device = next(net.parameters()).device
    total_entropy = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            names, images, masks = data

            images = images.to(model_device)
            masks = masks.to(model_device)

            outputs = net(images)

            for true, pred in zip(masks, outputs):
                logger.debug("Calculating entropy")
                pred = (pred > 0.5).float()
                entropy = F.binary_cross_entropy_with_logits(pred, true).item()
                logger.debug(f"{i:<5d}Calculated entropy between true and predicted masks: {entropy:.2f}")
                total_entropy += entropy

    logger.info(f"BCE Loss: {total_entropy}")
    return total_entropy
