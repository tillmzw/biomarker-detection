#!/usr/bin/env python3

import sys
import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import BinaryPatchIDRIDDataset
import model
import utils


def generate_cam(feature_conv, weight_softmax, classes=range(0, 5), resize=(256, 256)):
    batches, channels, h, w = feature_conv.shape
    # TODO: do i need softmax weights? resnet doesn't have softmax and i'm using sigmoids for the loss
    output = []
    for idx in classes:
        cam = weight_softmax[idx].dot(feature_conv.reshape((channels, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_res = cv2.resize(cam_img, resize)
        output.append(cam_res)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-s", "--state")
    parser.add_argument("-S", "--scratch")

    parser.add_argument("-t", "--use-training-dataset", action="store_true", default=False)
    parser.add_argument("-v", "--use-validation-dataset", action="store_true", default=True)
    parser.add_argument("-l", "--limit", default=None, type=int, help="Limit total number of images processed")
    parser.add_argument("-L", "--class-limit", default=None, type=int, help="Process this many images per class")

    args = parser.parse_args()

    # initialize model
    net = getattr(model, args.model)()
    net.load_state_dict(torch.load(args.state, map_location="cpu"))
    net.eval()

    # register hooks to model to record the output after a given layer
    features = []  # TODO: make sure this doesn't overflow and recreate it appropriately
    def module_forward_hook(module, input, output):
        """Run after the forward() pass of a given module and keeps track of its output."""
        sig = nn.Sigmoid()(output)
        features.append(sig.data.cpu().numpy())
        return None  # this makes this hook not affect the module

    net.resnet._modules.get("layer4").register_forward_hook(module_forward_hook)
    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    # initialize output dirs (root and one dir per class)
    if not args.scratch.endswith("cam"):
        args.scratch = os.path.join(args.scratch, "cam")
    for d in [args.scratch, *[os.path.join(args.scratch, str(cls)) for cls in range(len(BinaryPatchIDRIDDataset.CLASSES))]]:
        if not os.path.isdir(d) and os.path.exists(d):
            raise argparse.ArgumentError(f"{d} must be non-existent or a directory")
        elif not os.path.exists(d):
            os.mkdir(d)

    # initialize datasets
    datasets = []
    if args.use_training_dataset:
        datasets.append(BinaryPatchIDRIDDataset("train", limit=args.limit))
    if args.use_validation_dataset:
        datasets.append(BinaryPatchIDRIDDataset("test", limit=args.limit))

    dataloaders = [DataLoader(ds, batch_size=1) for ds in datasets]

    # iterate over data
    for dataloader in tqdm(dataloaders, unit="dataloader"):
        class_counters = torch.zeros((1, len(BinaryPatchIDRIDDataset.CLASSES)))
        for data in tqdm(dataloader, unit="image"):
            features = []
            names, coords, images, masks = data
            assert len(images) == 1
            outputs = net(images)
            class_counters += masks
            cams = generate_cam(features[0], weight_softmax)
            image = images[0, :, :, :]
            orig_image_path = dataloader.dataset.get_path_for_name(names[0])
            cv_image = utils.unnorm_transform(image)
            cv_image = cv_image.permute(1, 2, 0)
            cv_image *= 255
            cv_image = cv_image.detach().cpu().numpy()

            full_image = cv2.imread(orig_image_path)
            (row_start, col_start), (row_end, col_end) = coords

            _, height, width = image.shape
            for cls, cam in enumerate(cams):
                colormap = cv2.resize(cam, (width, height))
                heatmap = cv2.applyColorMap(colormap, cv2.COLORMAP_JET)
                fullcam = heatmap * 0.1 + cv_image * 0.5  # adapt pixel intensities to simulate overlay when merging

                heatmap_full = np.zeros(full_image.shape, dtype=np.uint8)
                heatmap_full[row_start:row_end, col_start:col_end, :] = heatmap
                merge = cv2.addWeighted(full_image, 1, heatmap_full, 0.6, 0)

                d = os.path.join(args.scratch, str(cls), f"{names[0]}_cam.jpg")
                d_full = os.path.join(args.scratch, str(cls), f"{names[0]}_full.jpg")
                d_merge = os.path.join(args.scratch, str(cls), f"{names[0]}_merge.jpg")

                cv2.imwrite(d, fullcam)
                cv2.imwrite(d_merge, merge)
                if not os.path.exists(d_full):
                    os.symlink(orig_image_path, d_full)

            if args.class_limit and class_counters.min() >= args.class_limit:
                print(f"Aborting loop since the smallest class has now {class_counters.min()} entries")
                break

    import time; time.sleep(1)