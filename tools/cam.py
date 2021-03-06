#!/usr/bin/env python3

import sys
import os
import argparse
import math

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


class CAMHook:
    def __init__(self, weights):
        self.feature_conv = None
        self.weights = weights

    def __call__(self, module, input, output):
        # Note: the sigmoid is applied during CAM generation
        self.feature_conv = output.data.cpu().numpy()

    def clear(self):
        self.feature_conv = None

    def generate_cams(self, classes=range(0, 5), resize=(256, 256)):
        batches, channels, h, w = self.feature_conv.shape
        output = []
        for idx in classes:
            class_weight = self.weights[idx]
            cam = class_weight.dot(self.feature_conv.reshape((channels, h * w)))
            cam = cam.reshape(h, w)
            #cam = cam - np.min(cam)
            #cam_img = cam / np.max(cam)
            # calculate the sigmoid - since this is not a tensor but a ndarray, we can't use torch.sigmoid()
            sig_cam = 1 / (1 + np.exp(-cam))
            cam_res = cv2.resize(sig_cam, resize)
            output.append(cam_res)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="ResNet50")
    parser.add_argument("-s", "--state")
    parser.add_argument("-S", "--scratch")

    parser.add_argument("-t", "--use-training-dataset", action="store_true", default=False)
    parser.add_argument("-v", "--use-validation-dataset", action="store_true", default=True)
    parser.add_argument("-l", "--limit", default=None, type=int, help="Limit total number of images processed")
    parser.add_argument("-L", "--class-limit", default=None, type=int, help="Process this many images per class")

    parser.add_argument("-d", "--discrepancies", action="store_true", default=False, help="Always keep samples where distance between prediction and truth is very large")
    parser.add_argument("-a", "--annotations", choices=("top", "file", "none"), default="top", help="Declare where image annotations will be placed")
    parser.add_argument("-c", "--context", type=int, default=-1, help="Retain this many pixels around sample as contextual information in output")

    args = parser.parse_args()

    if args.use_training_dataset:
        print(f"ATTENTION: Using the training dataset may produce rotated or otherwise transformed and hence unaligned CAMs.")

    # initialize model
    net = getattr(model, args.model)()
    net.load_state_dict(torch.load(args.state, map_location="cpu"))
    net.eval()
    # get the softmax weight
    params = list(net.parameters())
    # get the weights from the last (linear) layer
    weight_softmax = np.squeeze(params[-2].data.numpy())
    # build & register the hook

    cam_hook = CAMHook(weights=weight_softmax)
    net.resnet._modules.get("layer4").register_forward_hook(cam_hook)

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

    dataloaders = [DataLoader(ds, batch_size=1, shuffle=True) for ds in datasets]

    # iterate over data
    for dataloader in tqdm(dataloaders, unit="dataloader"):
        class_counters = torch.zeros((1, len(BinaryPatchIDRIDDataset.CLASSES)))
        for i, data in enumerate(tqdm(dataloader, unit="image")):
            names, coords, images, masks = data

            assert len(images) == 1  # additional assertion to make sure we're only working a single image
            (row_start, col_start), (row_end, col_end) = coords
            image = images[0, :, :, :]
            _, height, width = image.shape
            # double reconfirm that image dimensions match expectations
            assert (height, width) == ((col_end - col_start), (row_end - row_start))
            # DEBUG:
            #print(f"{name}: row: {row_start.item():>4} - {row_end.item():>4}; cols: {col_start.item():>4} - {col_end.item():>4}")
            class_counters += masks

            outputs = net(images)
            sig = torch.sigmoid(outputs)
            cams = cam_hook.generate_cams(resize=(height, width))
            cam_hook.clear()  # prep for next iteration

            # reconstruct the original input patch so that it's digestible for opencv
            rec_image = utils.unnorm_transform(image)
            rec_image = rec_image.permute(1, 2, 0)
            rec_image *= 255
            rec_image = rec_image.detach().cpu().numpy()

            orig_image_path = dataloader.dataset.get_path_for_name(names[0])
            full_image = cv2.imread(orig_image_path)

            for cls, cam in enumerate(cams):
                # should this class be positive?
                truth = masks[0][cls]
                true_class = truth == 1
                discrepancy = abs(truth - sig.flatten()[cls])
                is_massive_discrepancy = discrepancy > 0.95

                if is_massive_discrepancy and args.discrepancies:
                    # continue working here since this is a discrepancy and we want to plot them
                    pass
                elif true_class:
                    # continue working here since this a solid positive ground truth
                    pass
                else:
                    # either we don't care about discrepancies or this is a negative ground truth, so go to next class
                    continue

                class_name = BinaryPatchIDRIDDataset.CLASSES[cls]
                pretty_name = f"{names[0]}_{i:04d}_{class_name.lower().replace(' ', '-')}"

                if args.discrepancies and is_massive_discrepancy:
                    print(f"Massive prediction discrepancy detected in {pretty_name}: {discrepancy:.3f}")

                # data rescaling thanks to https://stackoverflow.com/a/56276534
                maxOld, minOld = cam.max(), cam.min()
                maxNew, minNew = cam.max() * 255, cam.min() * 255
                cam_rescaled = (maxNew - minNew) / (maxOld - minOld) * (cam - maxOld) + maxNew
                cam_rescaled = np.rint(cam_rescaled).astype(np.uint8)
                heatmap = cv2.applyColorMap(cam_rescaled, cv2.COLORMAP_JET)

                overlayed_cam = heatmap * 0.3 + rec_image * 0.9  # adapt pixel intensities to simulate overlay when merging

                # heatmap, resized to be the same size as the original image (4288x2488)
                heatmap_full = np.zeros(full_image.shape, dtype=np.uint8)
                # fill in the heatmap data
                heatmap_full[row_start:row_end, col_start:col_end, :] = heatmap
                # overlay heatmap over original image (with alpha values)
                merge = cv2.addWeighted(full_image, 1, heatmap_full, 0.6, 0)

                # manage contextual pixels around sample
                if args.context > -1:
                    # round up if necessary so we have even numbers
                    context = math.ceil(args.context / 2) * 2
                    c = context // 2
                    # dont crop outside of image bounds
                    rb_lower = 0
                    rb_upper = merge.shape[0]
                    cb_lower = 0
                    cb_upper = merge.shape[1]
                    merge = merge[
                            max(row_start - c, rb_lower): min(row_end + c, rb_upper),
                            max(col_start - c, cb_lower): min(col_end + c, cb_upper),
                            :]
                    crop_image = full_image[
                            max(row_start - c, rb_lower): min(row_end + c, rb_upper),
                            max(col_start - c, cb_lower): min(col_end + c, cb_upper),
                            :]
                    if args.annotations != "file":
                        print("Overwriting annotation style to \"file\" due to space constraints")
                    args.annotations = "file"
                else:
                    crop_image = full_image

                # add some textual information to the output image
                if args.discrepancies and is_massive_discrepancy:
                    addendum = f"[W! Discrepancy detected: {discrepancy:.3f}]"
                else:
                    addendum = ""
                title = f"Detector: {BinaryPatchIDRIDDataset.CLASSES[cls]} {addendum}"
                descs = []
                for i in range(5):
                    # mark a class as *a*bsent or *p*resent
                    mod = {0: "a", 1: "p"}[masks[0][i].round().item()]
                    descs.append(f"{BinaryPatchIDRIDDataset.CLASSES[i]} [{mod}]: {sig[0][i]:.2f}")

                if args.annotations == "top":
                    cv2.putText(merge, title, (10, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(128, 255, 128), thickness=2)
                    cv2.putText(merge, "; ".join(descs), (10, 200), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(128, 255, 128), thickness=1)
                elif args.annotations == "file":
                    with open(os.path.join(args.scratch, str(cls), f"{pretty_name}_annot.txt"), "w") as af:
                        af.write(title)
                        for d in descs:
                            af.write("\n\t" + d)

                # save all these images to disk.
                d = os.path.join(args.scratch, str(cls), f"{pretty_name}_cam.jpg")
                d_full = os.path.join(args.scratch, str(cls), f"{pretty_name}_full.jpg")
                d_merge = os.path.join(args.scratch, str(cls), f"{pretty_name}_merge.jpg")
                d_crop = os.path.join(args.scratch, str(cls), f"{pretty_name}_context.jpg")

                cv2.imwrite(d, overlayed_cam)
                cv2.imwrite(d_merge, merge)

                if args.context > -1:
                    cv2.imwrite(d_crop, crop_image)

                # for the original image a symlink should be sufficient
                if not os.path.exists(d_full):
                    os.symlink(orig_image_path, d_full)

            if args.class_limit and class_counters.min() >= args.class_limit:
                print(f"Aborting loop since the smallest class has now {class_counters.min()} entries")
                break
