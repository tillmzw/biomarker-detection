#!/usr/bin/env python3

import sys
import os
import argparse
from tqdm import tqdm
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import BinaryPatchIDRIDDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training-ds", action="store_true", default=False, help="Use training dataset")
    parser.add_argument("-T", "--testing-ds", action="store_true", default=False, help="Use testing dataset")
    parser.add_argument("-p", "--patch-size", type=int, default=500, help="Split images into chunks of this size")
    parser.add_argument("-P", "--presence-threshold", type=int, default=10, help="Require this many pixels of a class in a sample to consider it positive")
    parser.add_argument("-n", "--patch-number", type=int, default=100, help="Patch number per image")

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


    counts = [0 for _ in BinaryPatchIDRIDDataset.CLASSES]
    total = sum(map(lambda i: len(i), datasets))

    for ds in tqdm(datasets, unit="dataset"):
        for data in tqdm(ds, unit="image"):
            _, _, _, masks = data

            for k, (prev, this) in enumerate(zip(counts, masks)):
                counts[k] = prev + this

        print()

    header_fmt  = "{label:^15} {abs_c:>8} {prop_c:>12} {rel_c:>12}"
    content_fmt = "{label:^15} {abs_c:>8} {prop_c:>12.3f} {rel_c:>12.3f}"
    header = header_fmt.format(**{
        "label": "label",
        "abs_c": "count",
        "prop_c": "prop.",
        "rel_c": "rel."
    })
    header_desc = header_fmt.format(**{
        "label": "",
        "abs_c": "",
        "prop_c": "count/positives",
        "rel_c": "count/total"
    })
    print(header)
    print(header_desc)
    print(len(header) * "-")

    for label, count in zip(BinaryPatchIDRIDDataset.CLASSES, counts):
        if isinstance(count, torch.Tensor):
            abs_c = count.item()
        else:
            abs_c = count
        rel_c = abs_c / total
        if sum(counts) == 0:
            prop_c = 0.0
        else:
            prop_c = abs_c / sum(counts)
        print(content_fmt.format(**{"label": label, "abs_c": abs_c, "prop_c": prop_c, "rel_c": rel_c}))

    print(len(header) * "=")
    print(content_fmt.format(**{"label": "", "abs_c": sum(counts), "prop_c": 1, "rel_c": (total / sum(counts))}))
    print(len(header) * ".")
    print(header_fmt.format(**{"label": "# images", "abs_c": total, "prop_c": "", "rel_c": ""}))
