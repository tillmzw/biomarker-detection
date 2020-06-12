#!/usr/bin/env python3

import os
import csv
import logging
import itertools
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, average_precision_score, precision_recall_curve

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

        # calculate precision recall curves - needs special processing since this is only defined for a binary case
        # so we handle every class separately;
        # see https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        logger.debug("Calculating precision-recall curves")
        precision = {}
        recall = {}
        avg_precision_class = {}
        for c_idx in range(5):
            truth_c = ground_truth[:, c_idx].to("cpu")
            pred_c = predictions[:, c_idx].to("cpu")
            precision[c_idx], recall[c_idx], _ = precision_recall_curve(y_true=truth_c, probas_pred=pred_c)
            avg_precision_class[c_idx] = average_precision_score(y_true=truth_c, y_score=pred_c)

        prc = (precision, recall, avg_precision_class)

        return acc, loss, avg_precision, confusion, prc

    finally:
        if record_file:
            record_file.close()
