#!/usr/bin/env python3

import sys
import os
import argparse
import multiprocessing
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import BinaryPatchIDRIDDataset


def print_counts(counts, n_images, title=None):
    header_fmt = "{label:^15} {abs_c:>8} {prop_c:>12} {rel_c:>12} {weights_c:>12}"
    content_fmt = "{label:^15} {abs_c:>8} {prop_c:>12.3f} {rel_c:>12.3f} {weights_c:>12.3f}"
    header = header_fmt.format(
        label="label",
        abs_c="count",
        prop_c="prop.",
        rel_c="rel.",
        weights_c="weights"
    )
    header_desc = header_fmt.format(
        label="",
        abs_c="",
        prop_c="count/positives",
        rel_c="count/total",
        weights_c=""
    )
    print()
    if title:
        print(title)
    print(header)
    print(header_desc)
    print(len(header) * "-")

    for label, count in zip(BinaryPatchIDRIDDataset.CLASSES, counts):
        if isinstance(count, torch.Tensor):
            abs_c = count.item()
        else:
            abs_c = count
        rel_c = abs_c / n_images
        if sum(counts) == 0:
            prop_c = 0.0
        else:
            prop_c = abs_c / sum(counts)
        # weight calculation follows exactly
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        if abs_c > 0:
            weight = n_images / (len(BinaryPatchIDRIDDataset.CLASSES) * abs_c)
        else:
            weight = float('nan')

        print(content_fmt.format(label=label, abs_c=abs_c, prop_c=prop_c, rel_c=rel_c, weights_c=weight))

    print(len(header) * "=")
    print(content_fmt.format(label="", abs_c=sum(counts), prop_c=1, rel_c=(n_images / sum(counts)), weights_c=float("nan")))
    print(len(header) * ".")
    print(header_fmt.format(label="# images", abs_c=n_images, prop_c="", rel_c="", weights_c=""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training-ds", action="store_true", default=False, help="Use training dataset")
    parser.add_argument("-T", "--testing-ds", action="store_true", default=False, help="Use testing dataset")
    parser.add_argument("-p", "--patch-size", type=int, default=500, help="Split images into chunks of this size")
    parser.add_argument("-P", "--presence-threshold", type=int, default=10, help="Require this many pixels of a class in a sample to consider it positive")
    parser.add_argument("-n", "--patch-number", type=int, default=100, help="Patch number per image")
    parser.add_argument("-c", "--cpus", type=int, default=multiprocessing.cpu_count(), metavar=multiprocessing.cpu_count(), help="Max. number of CPUs used for processing")
    parser.add_argument("--plots", nargs="?", const=False, help="Generate a matplotlib plot for the distributions, optionally saving it in a file.")

    args = parser.parse_args()

    datasets = []
    if args.training_ds:
        ds = BinaryPatchIDRIDDataset("train",
                                     patch_size=args.patch_size,
                                     n_patches=args.patch_number,
                                     presence_threshold=args.presence_threshold)
        datasets.append(ds)
    if args.testing_ds:
        ds = BinaryPatchIDRIDDataset("test",
                                     patch_size=args.patch_size,
                                     n_patches=args.patch_number,
                                     presence_threshold=args.presence_threshold)
        datasets.append(ds)

    assert datasets, "No datasets provided"

    for arg in vars(args):
        print("%18s: %s" % (arg, getattr(args, arg)))

    total_counts = [0 for _ in BinaryPatchIDRIDDataset.CLASSES]
    datasets_counts = [[0 for _ in BinaryPatchIDRIDDataset.CLASSES] for _ in datasets]

    for i, ds in enumerate(tqdm(datasets, unit="dataset")):
        class_counts = [0 for _ in BinaryPatchIDRIDDataset.CLASSES]
        # using a dataloader here speeds everything up due to the multiprocessing capabilities!
        dl = torch.utils.data.dataloader.DataLoader(ds, num_workers=args.cpus, batch_size=1)
        for data in tqdm(dl, unit="image"):
            _, _, _, masks = data
            # since we use batch_size = 1, we can flatten here (2nd dim empty due to single sample)
            masks = masks.flatten()
            for k, (prev, this) in enumerate(zip(class_counts, masks)):
                class_counts[k] = (prev + this).item()
        # add this iteration's class counts to the total counts, element-wise
        total_counts = [sum(x) for x in zip(total_counts, class_counts)]
        datasets_counts[i] = class_counts.copy()

    # RESULTS
    # print totals first (if different from results for a specific dataset)
    if len(datasets) > 1:
        # sum length of all datasets
        n_images = sum(map(lambda i: len(i), datasets))
        print_counts(total_counts, n_images, "Total Counts")

    # print dataset specifics after
    for i, d_counts in enumerate(datasets_counts):
        d_type = datasets[i]._itype.capitalize()
        d_title = f"{d_type} Counts"
        print_counts(d_counts, n_images=len(datasets[i]), title=d_title)

    if args.plots is not None:
        fig, (ax_prop_pos, ax_prop_total) = plt.subplots(2, 1, sharex='col', figsize=(7, 8))
        w = 0.2
        x = np.arange(5)
        cmap = cm.get_cmap("Set2")
        ax_prop_pos.set_title("Proportion of biomarkers")
        ax_prop_total.set_title("Presence in samples")

        for i, counts in enumerate(datasets_counts):
            y_prop_pos = np.array(counts) / len(datasets[i])
            y_prop_total = np.array(counts) / sum(counts)

            # Note: offset doesn't work for > 2 datasets
            offset = w/2 * (-1 if i == 0 else 1)
            b_pos = ax_prop_pos.bar(x + offset, y_prop_pos, width=w, color=cmap.colors[i])
            b_tot = ax_prop_total.bar(x + offset, y_prop_total, width=w, color=cmap.colors[i])

            d_type = datasets[i]._itype.capitalize()
            b_pos.set_label(d_type)
            b_tot.set_label(d_type)

        for ax in (ax_prop_pos, ax_prop_total):
            ax.set_ylabel("%")
            ax.set_xticks(x)
            ax.set_xticklabels(datasets[0].CLASSES, rotation=45)
            ax.legend()
        if args.plots is not False:
            # args.plots is a file
            plt.savefig(args.plots)
        else:
            # no file. display
            plt.show()
