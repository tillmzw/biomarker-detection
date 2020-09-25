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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model
from dataset import BinaryPatchIDRIDDataset
from validator import validate


def filename(name, prefix="", postfix=""):
    ap = os.path.abspath(name)
    if os.path.isdir(name):
        raise RuntimeError(f"\"{name}\" is a directory - need a file")
    dir = os.path.dirname(ap)
    fn = os.path.basename(ap)
    f, ext = os.path.splitext(fn)
    return os.path.join(dir, f"{prefix}{f}{postfix}{ext}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=model.list_models()[0], choices=model.list_models())
    parser.add_argument("-s", "--state", default="./model.pth", help="Model state file")
    parser.add_argument("-b", "--batch", type=int, default=24, metavar=24)
    parser.add_argument("-l", "--limit", type=float, default=None)
    parser.add_argument("-D", "--device", choices=("cpu", "cuda", "auto"), default="auto", help="Run on this device")
    parser.add_argument("-p", "--patch-size", type=int, default=500, help="Split images into chunks of this size")
    parser.add_argument("-n", "--patch-number", type=int, default=50, help="Patch number per image.")
    parser.add_argument("-c", "--cpus", type=int, default=multiprocessing.cpu_count(), metavar=multiprocessing.cpu_count(), help="Max. number of CPUs used for processing")
    parser.add_argument("--plots", nargs="?", const=False, help="Generate a matplotlib plot for the distributions, optionally saving it in a file.")
    parser.add_argument("-P", "--presence-threshold", nargs="*", type=int, default=[10], help="Require this many pixels of a class in a sample to consider it positive")

    args = parser.parse_args()
    for arg in vars(args):
        print("%18s: %s" % (arg, getattr(args, arg)))

    assert os.path.isfile(args.state), f"State file not found: {args.state}"

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hasattr(model, args.model):
        raise parser.error("Model \"%s\" is unknown; available options: %s" % (args.model, ", ".join(model.__all__)))
    net = getattr(model, args.model)()
    net.to(torch.device(args.device))
    net.load_state_dict(torch.load(args.state, map_location=torch.device(args.device)))

    rocs = [[0 for _ in args.presence_threshold] for _ in BinaryPatchIDRIDDataset.CLASSES]
    for t_iter, t in enumerate(tqdm(args.presence_threshold, unit="threshold configuration")):
        testset = BinaryPatchIDRIDDataset("test",
                                 limit=args.limit,
                                 patch_size=args.patch_size,
                                 n_patches=args.patch_number,
                                 presence_threshold=t)

        testloader = DataLoader(testset, batch_size=args.batch, num_workers=args.cpus, shuffle=True, pin_memory=True)

        acc, loss, avg_precision, confusion, prc, roc = validate(net, testloader)
        fpr, tpr, roc_auc = roc

        # every biomarker gets n entries - one for each threshold. flip data hierarchy
        for biomarker_index in roc_auc.keys():
            rocs[biomarker_index][t_iter] = roc_auc[biomarker_index]

    fig, ax = plt.subplots(1, 1)
    x = np.arange(len(args.presence_threshold))

    cmap = cm.get_cmap("Set2")

    for i, (biomarker_name, roc) in enumerate(zip(BinaryPatchIDRIDDataset.CLASSES, rocs)):
        ax.plot(x, roc, label=biomarker_name, color=cmap.colors[i])

    ax.set_title("AUC in relation to pixel thresholds")
    ax.set_ylabel("AUC")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Presence Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(args.presence_threshold)
    ax.legend()

    plt.tight_layout()
    if args.plots is not False:
        # args.plots is a file
        fname = filename(args.plots)
        plt.savefig(fname)
    else:
        # no file. display
        plt.show()


