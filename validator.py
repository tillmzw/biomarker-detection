#!/usr/bin/env python3

import os
import csv
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, multilabel_confusion_matrix

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

    # TODO: is there a better way to do this?
    predictions = torch.tensor(data=(), dtype=torch.float).to(model_device)
    truth = torch.tensor(data=(), dtype=torch.float).to(model_device)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            names, images, masks = data

            images = images.to(model_device)
            masks = masks.to(model_device)

            outputs = net(images)

            predictions = torch.cat((predictions, outputs))
            #predictions = torch.cat((predictions, predicted))
            truth = torch.cat((truth, masks))

    # as predictions are still float values (as given by a sigmoid function), round them first
    predictions = torch.round(predictions)
    # .eq() does element-wise equality checks
    # Note: .eq() != .equal()
    overlap = torch.eq(truth, predictions)
    # accuracy: fraction of matching predictions over total item number
    # `.item()` on a single-item tensor to extract the value
    acc = torch.sum(overlap).item() / truth.numel() * 100

    confusion = multilabel_confusion_matrix(y_true=truth.to("cpu"), y_pred=predictions.to("cpu"))

    # FIXME: fix kappa calculation
    # kappa = quadratic_kappa(predictions, truth)
    kappa = 0
    return acc, kappa, confusion
