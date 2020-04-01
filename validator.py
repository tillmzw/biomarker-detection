#!/usr/bin/env python3

import os
import csv
import logging
import itertools
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, average_precision_score

logger = logging.getLogger(__name__)


def validate(net, dataloader, record_filename=None, loss_func=None):
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
            logger.warning(f"Overwriting record log: {record_filename}")
        record_file = open(record_filename, "w")
        f = ("name",
             "top_left",
             "bottom_right",
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
                names, coords, images, masks = data

                images = images.to(model_device)
                masks = masks.to(model_device)

                outputs = net(images)

                if record_file and record_writer:
                    # at this point, I expect a meta to have 3 elements; if that's not true, the used
                    # dataset is not matching that expectation
                    for i, (truth, prediction) in enumerate(zip(masks, outputs)):
                        # NOTE: due to dataloader batching/stacking, the dimensions are rotated and items are tensors.
                        row = {"name": names[i],
                               "top_left": "%s, %s" % (coords[0][0][i].item(), coords[0][1][i].item()),
                               "bottom_right": "%s, %s" % (coords[1][0][i].item(), coords[1][1][i].item()),
                                }
                        for j, (t, p) in enumerate(zip(truth, prediction)):
                            row[f"truth_{j}"] = t.item()
                            row[f"prediction_{j}"] = p.item()
                        record_writer.writerow(row)

                predictions = torch.cat((predictions, outputs))
                ground_truth = torch.cat((ground_truth, masks))

        # as predictions are still float values (as given by a sigmoid function), round them first
        predictions_rounded = torch.round(predictions)
        # .eq() does element-wise equality checks
        # Note: .eq() != .equal()
        overlap = torch.eq(ground_truth, predictions_rounded)
        # accuracy: fraction of matching predictions over total item number
        # `.item()` on a single-item tensor to extract the value
        acc = torch.sum(overlap).item() / ground_truth.numel() * 100

        if loss_func:
            loss = loss_func(predictions, ground_truth)
        else:
            loss = None

        confusion = multilabel_confusion_matrix(y_true=ground_truth.to("cpu"), y_pred=predictions_rounded.to("cpu"))

        # calculate average precision scores
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        # FIXME: this returns `nan` (as guard against division by zero) if all ground_truths are zero.
        avg_precision = average_precision_score(y_true=ground_truth.to("cpu"), y_score=predictions.to("cpu"))

        return acc, loss, avg_precision, confusion

    finally:
        if record_file:
            record_file.close()
