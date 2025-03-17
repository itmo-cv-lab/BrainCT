"""Segmentation models definition"""

import segmentation_models_pytorch as smp
import torch
from torch import nn


class SegmentationModel(nn.Module):
    """Segmentation model"""

    def __init__(
        self, architecture="FPN", encoder="efficientnet-b0", classes=2, pretrained=True
    ):
        super(SegmentationModel, self).__init__()

        arch = architecture.lower()

        if arch == "fpn":
            self.model = smp.FPN(
                encoder_name=encoder,
                classes=classes,
                in_channels=3,
                encoder_weights="imagenet" if pretrained else None,
            )
        elif arch == "unet":
            self.model = smp.Unet(
                encoder_name=encoder,
                classes=classes,
                in_channels=3,
                encoder_weights="imagenet" if pretrained else None,
            )
        elif arch == "unetplusplus":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder,
                classes=classes,
                in_channels=3,
                encoder_weights="imagenet" if pretrained else None,
            )
        elif arch == "pan":
            self.model = smp.PAN(
                encoder_name=encoder,
                classes=classes,
                in_channels=3,
                encoder_weights="imagenet" if pretrained else None,
            )
        elif arch == "deeplabv3":
            self.model = smp.DeepLabV3(
                encoder_name=encoder,
                classes=classes,
                in_channels=3,
                encoder_weights="imagenet" if pretrained else None,
            )
        elif arch == "segformer":
            self.model = smp.MAnet(
                encoder_name=encoder,
                classes=classes,
                in_channels=3,
                encoder_weights="imagenet" if pretrained else None,
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x):
        """Forward method"""

        return self.model(x)


class FullModel(nn.Module):
    """Combine preprocessing and segmentation models"""
    def __init__(self, preprocessing_model, segmentation_model):
        super(FullModel, self).__init__()
        self.preprocessing_model = preprocessing_model
        self.segmentation_model = segmentation_model

    def forward(self, x):
        """Forward method"""

        enhanced_image = self.preprocessing_model(x)
        segmentation_output = self.segmentation_model(enhanced_image)

        return segmentation_output
