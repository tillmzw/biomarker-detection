#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import multiprocessing
import tempfile

import torch
import wandb
from torch.utils.data import DataLoader
import tabulate

import model
from dataset import BinaryPatchIDRIDDataset
import training
import utils
from validator import validate


logger = logging.getLogger(__name__)
CPU_COUNT = min(4, multiprocessing.cpu_count())

logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":

    PARENT_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--state", type=str, default=None)
    parser.add_argument("-b", "--batch", type=int, default=24, metavar=24)
    parser.add_argument("-e", "--epochs", type=int, default=2, metavar=2)
    parser.add_argument("-d", "--dir", type=str, default=PARENT_DIR, metavar=PARENT_DIR, help="base dir")
    parser.add_argument("--data-dir", default=os.path.join(PARENT_DIR, "data"), metavar=os.path.join(PARENT_DIR, "data"), help="Path to directory containing the image sets")
    parser.add_argument("-l", "--limit", type=float, default=None, help="Limit training dataset to this many entries; can be an integer (number of samples) or a float (fraction of samples). Requires --train")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-t", "--train", action="store_true", default=False, help="Run training phase")
    parser.add_argument("-m", "--model", default=model.list_models()[0], choices=model.list_models())
    parser.add_argument("-D", "--device", choices=("cpu", "cuda", "auto"), default="auto", help="Run on this device")
    parser.add_argument("-V", "--validate", action="store_true", default=False, help="Run validation tests")
    parser.add_argument("-L", "--validation-limit", type=float, default=None, help="During validation, limit validation set to this number of samples; can be an integer (number of samples) or a float (fraction of samples). Requires --validation")
    parser.add_argument("-S", "--scratch", default=None, help="Directory to place additional outputs in")
    parser.add_argument("-M", "--mismatches", action="store_true", default=False, help="Record mismatches between predictions and ground truth; requires scratch directory. Slows operation!")
    parser.add_argument("-p", "--patch-size", type=int, default=256, help="Split images into chunks of this size")
    parser.add_argument("-P", "--presence-threshold", type=int, default=100, help="Require this many pixels of a class in a sample to consider it positive")
    parser.add_argument("--log", default=None, help="Write all log file to this file")
    parser.add_argument("-N", "--no-wandb", action="store_true", default=False, help="Dont send results to wandb")
    # TODO: support >1 GPU

    args = parser.parse_args()
    del PARENT_DIR

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handlers = (logging.StreamHandler(sys.stdout),)

    if args.log:
        # write to a file and abort if that file exists already
        file_handler = logging.FileHandler(filename=os.path.join(args.dir, args.log), mode="x")
        handlers += (file_handler,)

    # attach the formatter to the different handlers, set log levels, and attach handlers to root logger
    logging.basicConfig(level=level,
                        format=('%(asctime)s %(levelname)8s %(name)10s %(lineno)3d -- %(message)s'),
                        datefmt="%H:%M:%S",
                        handlers=handlers)

    # initialize early so that the wandb logging handlers are attached
    wandb_cfg = {"project": "biomarker_detection"}
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(**wandb_cfg)

    logger.info("Command line arguments:")
    for arg in vars(args):
        logger.info("%18s: %s" % (arg, getattr(args, arg)))

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.scratch and not os.path.exists(args.scratch):
        logger.info("Creating directory %s" % args.scratch)
        os.makedirs(args.scratch)

    if not hasattr(model, args.model):
        raise parser.error("Model \"%s\" is unknown; available options: %s" % (args.model, ", ".join(model.__all__)))
    net = getattr(model, args.model)()

    net.to(torch.device(args.device))
    data_dir = args.data_dir

    if args.validate:
        testset = BinaryPatchIDRIDDataset("test", path=data_dir, limit=args.validation_limit, patch_size=args.patch_size, presence_threshold=args.presence_threshold)
        testloader = DataLoader(testset, batch_size=args.batch, num_workers=CPU_COUNT, shuffle=True)
    else:
        testloader = None

    if args.train:
        logger.info("Starting training")

        trainset = BinaryPatchIDRIDDataset("train", path=data_dir, limit=args.limit, patch_size=args.patch_size, presence_threshold=args.presence_threshold)
        trainloader = DataLoader(trainset, batch_size=args.batch, num_workers=CPU_COUNT, shuffle=True)

        trainer = training.AdamTrainer(epochs=args.epochs)
        trainer.train(net, trainloader, args.state, validation_dataloader=testloader)
    else:
        if not args.state:
            raise RuntimeError("Need a state file if training is skipped")
        if not os.path.isfile(args.state):
            raise RuntimeError("State \"%s\" is not a file" % args.state)
        logger.info("Loading model state from %s" % args.state)
        net.load_state_dict(torch.load(args.state, map_location=torch.device(args.device)))

    if args.validate:
        # mismatch record file
        mmr = os.path.join(args.scratch, "mismatches.csv") if (args.scratch and args.mismatches) else None
        acc, kappa, confusion = validate(net, testloader, record_file=mmr)
        # log the confusion matrix
        logger.info("Final validation run: %05.2f%% accuracy, kappa = % 04.2f" % (acc, kappa))

        """
        # FIXME: this generates a floating point exception (SIGFPE) when run on
        #  multilabel resnets and without a proper confusion matrix. 
        logger.info("Confusion matrix:")
        confusion_norm = utils.norm_mat(confusion, norm="rows")
        # because we plot the indices, the first header item should be empty for clarity
        headers = (" ",) + tuple(range(confusion.shape[0] + 1))
        table = tabulate.tabulate(confusion_norm, headers, showindex="always", tablefmt="fancy_grid")
        for line in table.split("\n"):
            logger.info(line)
        """
        # create a plot from the confusion matrix
        logger.info("Creating confusion matrix plot")
        plot = utils.plot_confusion_matrix(confusion)
        implot = utils.plot_to_pil(plot)
        if args.scratch:
            utils.save_pil_to_scratch(implot, target_dir=args.scratch, name="final_confusion.png", overwrite=True)
        wandb.log({"confusion_matrix": wandb.Image(implot)})
