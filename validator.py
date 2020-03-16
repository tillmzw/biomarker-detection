#!/usr/bin/env python3

import os
import csv
import logging
import itertools
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, multilabel_confusion_matrix

logger = logging.getLogger(__name__)


def validate(net, dataloader, record_filename=None):
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
    ground_truth = torch.tensor(data=(), dtype=torch.float).to(model_device)

    if record_filename:
        if os.path.exists(record_filename):
            logger.warning(f"Overwriting record log: f{record_filename}")
        record_file = open(record_filename, "w")
        f = ("name",
             "image_idx",
             "patch_idx",
             *(f"{type}_{i}" for i, type in itertools.product(range(5), ("truth", "prediction")))
             )
        record_writer = csv.DictWriter(record_file, fieldnames=f)
        record_writer.writeheader()
    else:
        record_file = None
        record_writer = None

    try:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                metas, images, masks = data

                images = images.to(model_device)
                masks = masks.to(model_device)

                outputs = net(images)

                if record_file and record_writer:
                    # at this point, I expect a meta to have 3 elements; if that's not true, the used
                    # dataset is not matching that expectation
                    for i, (truth, prediction) in enumerate(zip(masks, outputs)):
                        # because the dimensions are flipped and this is an oldschool python list, I have to resort
                        # to this - I think
                        name, image_idx, patch_idx = metas[0][i], metas[1][i], metas[2][i]
                        row = {"name": name,
                               "image_idx": image_idx.item(),
                               "patch_idx": patch_idx.item()}
                        for j, (t, p) in enumerate(zip(truth, prediction)):
                            row[f"truth_{j}"] = t.item()
                            row[f"prediction_{j}"] = p.item()
                        record_writer.writerow(row)

                predictions = torch.cat((predictions, outputs))
                # predictions = torch.cat((predictions, predicted))
                ground_truth = torch.cat((ground_truth, masks))

        # as predictions are still float values (as given by a sigmoid function), round them first
        predictions = torch.round(predictions)
        # .eq() does element-wise equality checks
        # Note: .eq() != .equal()
        overlap = torch.eq(ground_truth, predictions)
        # accuracy: fraction of matching predictions over total item number
        # `.item()` on a single-item tensor to extract the value
        acc = torch.sum(overlap).item() / ground_truth.numel() * 100

        confusion = multilabel_confusion_matrix(y_true=ground_truth.to("cpu"), y_pred=predictions.to("cpu"))

        return acc, confusion

    finally:
        if record_file:
            record_file.close()
