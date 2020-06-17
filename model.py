#!/usr/bin/env python3

import logging
import torch
import torch.nn as nn
from torch.functional import F
import torchvision.models as models


logger = logging.getLogger(__name__)


def list_models():
    # Note that the first element here might be used as default
    return "ResNet50", "ResNet101", "WideResNet50", "DenseNet161"


class ResNet(nn.Module):
    def __init__(self, name_fmt="resnet{depth}", depth=50, pretrained=True, freeze_backbone=False):
        super().__init__()
        self._depth = depth
        self._pretrained = pretrained
        self._freeze_backbone = freeze_backbone
        self._name = name_fmt.format(depth=depth)

        assert hasattr(models, self._name), "Model \"%s\" not found" % self._name
        logger.info(f"Loading model {self._name} (pretrained = {self._pretrained}, freeze backbone = {self._freeze_backbone})")
        self.resnet = getattr(models, self._name)(pretrained=self._pretrained, progress=False)

        if self._freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Feedback from Pablo:
        # instead of using additional linear layers to reduce number of classes, just replace the last layer.
        # Note: This *will* fail if the resnet is not using the bottleneck block (all resnets with d>=50 use bottleneck)
        #       since their last layer has fewer elements (512 * 1 instead of 512 * 4).
        self.resnet.fc = nn.Linear(512 * 4, 5)

    @property
    def descriptor(self):
        # used for tracking this model's performance. Currently used to update wandb
        return {"network": self._name, "pretrained": self._pretrained, "frozen_backbone": self._freeze_backbone}

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet50(ResNet):
    def __init__(self, pretrained=True):
        super().__init__(depth=50, pretrained=pretrained)


class ResNet101(ResNet):
    def __init__(self, pretrained=True):
        super().__init__(depth=101, pretrained=pretrained)


class WideResNet50(ResNet):
    def __init__(self, pretrained=True):
        super().__init__(name_fmt="wide_resnet{depth}_2", depth=50, pretrained=pretrained)


class DenseNet161(nn.Module):
    def __init__(self, pretrained=True, transfer=False):
        super().__init__()
        self._name = "densenet161"
        self._pretrained = pretrained
        self._transfer = transfer

        logger.info("Loading model %s (pretrained = %s, transfer = %s)" % (self._name, pretrained, transfer))
        self.model = getattr(models, self._name)(pretrained=pretrained, progress=False)

        if transfer:
            logger.info("Disabling gradients on all feature-extracting layers")
            self.models.features.requires_grad = False

        # FIXME: This will always result in an uninitialized last FC
        # TODO: the model will return a 24 x 2208 tensor - here we expect 1000 x 5
        #       --> check how the in_features number is calculated and copy
        self.classifier = nn.Linear(in_features=1000, out_features=5)

    @property
    def descriptor(self):
        # used for tracking this model's performance. Currently used to update wandb
        return {"network": self._name, "pretrained": self._pretrained, "transfer": self._transfer}

    def forward(self, x):
        #features = self.model.features(x)
        #out = F.relu(features, inplace=True)
        #out = F.adaptive_avg_pool2d(out, (1, 1))
        #out = torch.flatten(out, 1)
        out = self.model.forward(x)
        out = self.classifier(out)
        return out
