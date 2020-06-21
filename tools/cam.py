import sys
import os
import argparse

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import BinaryPatchIDRIDDataset
import model


def generate_cam(feature_conv, classes=range(0, 5), resize=(100, 100)):
    batches, channels, h, w = feature_conv.shape
    # TODO: do i need softmax weights? resnet doesn't have softmax and i'm using sigmoids for the loss
    output = []
    for idx in classes:
        #cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = feature_conv.reshape(h, w)
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

    # register hooks to model
    # TODO: make sure this doesn't overflow and recreate it appropriately
    features = []
    def module_forward_hook(module, input, output):
        """Run after the forward() pass of a given module and keeps track of its output."""
        sig = F.sigmoid(output)
        features.append(sig.data.cpu().numpy())
        return None  # this makes this hook not affect the module

    net._modules.get("layer4").register_forward_hook(module_forward_hook)
    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    # initialize output dirs (root and one dir per class)
    for d in [args.scratch, *[os.path.join(args.scratch, str(cls)) for cls in range(BinaryPatchIDRIDDataset.CLASSES)]]:
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
    for dataloader in tqdm(dataloaders):
        class_counters = [0 for _ in range(BinaryPatchIDRIDDataset.CLASSES)]
        for data in tqdm(dataloader):
            features = []
            names, coords, images, masks = data
            assert len(images) == 1
            outputs = net(images)
            import pdb; pdb.set_trace()
            cams = generate_cam(features[0])
            image = images[0, :, :, :]
            cv_image = image.permute(1, 2, 0).detach().cpu().numpy()
            _, height, width = image.shape
            for cam in cams:
                colormap = cv2.resize(cam, (width, height))
                heatmap = cv2.applyColorMap(colormap, cv2.COLORMAP_JET)
                fullcam = heatmap * 0.3 + cv_image * 0.5

