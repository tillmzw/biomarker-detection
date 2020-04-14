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
    def __init__(self, name_fmt="resnet{depth}", depth=50, pretrained=True, transfer=False):
        super().__init__()
        self._depth = depth
        self._pretrained = pretrained
        self._transfer = transfer
        self._name = name_fmt.format(depth=depth)

        assert hasattr(models, self._name), "Model \"%s\" not found" % self._name
        logger.info("Loading model %s (pretrained = %s, transfer = %s)" % (self._name, pretrained, transfer))
        self.model = getattr(models, self._name)(pretrained=pretrained, progress=False)
        if transfer:
            logger.info("Disabling gradients on all feature-extracting layers")
            for param in self.model.parameters():
                param.requires_grad = False
        # disable the FC and replace it with a separate FC in the classifier attribute
        self.model.fc = Identity()
        self.features = self.model
        self.classifier = nn.Linear(in_features=512 * 4, out_features=5)

    @property
    def descriptor(self):
        # used for tracking this model's performance. Currently used to update wandb
        return {"network": self._name, "pretrained": self._pretrained, "transfer": self._transfer}

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.softmax(x)
        return x


class ResNet50(ResNet):
    def __init__(self, pretrained=True, transfer=False):
        super().__init__(depth=50, pretrained=pretrained, transfer=transfer)


class ResNet101(ResNet):
    def __init__(self, pretrained=True, transfer=False):
        super().__init__(depth=101, pretrained=pretrained, transfer=transfer)


class WideResNet50(ResNet):
    def __init__(self, pretrained=True, transfer=False):
        super().__init__(name_fmt="wide_resnet{depth}_2", depth=50, pretrained=pretrained, transfer=transfer)


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


class Identity(nn.Module):
    """
    A module that returns its input as output.
    Useful to functionally remove other modules.

    https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

