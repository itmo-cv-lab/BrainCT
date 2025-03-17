"""Enhancement models definition"""

import torch
from torch import nn
import torch.nn.functional as F

from enhancement_filters import EnhancementFilters


class EnhancementModel(nn.Module):
    """Enhancement model"""

    def __init__(
        self,
        input_channels,
        output_channels,
        fc_input_channels,
        fc_output_channels,
        parameters_foo,
        parameters_amounts,
        conv_block,
        norm_layer=nn.InstanceNorm2d,
        act_layer="lrelu",
        image_size=16,
    ):
        super(EnhancementModel, self).__init__()
        self.image_size = image_size
        self.conv_blocks = nn.Sequential()
        self.conv_blocks.add_module(
            "convblock1",
            conv_block(
                in_channels=input_channels[0],
                out_channels=output_channels[0],
                norm_layer=None,
                act_layer=act_layer,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
        )

        self.conv_blocks.add_module(
            "convblock2",
            conv_block(
                in_channels=input_channels[1],
                out_channels=output_channels[1],
                norm_layer=norm_layer,
                act_layer=act_layer,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
        )

        self.conv_blocks.add_module(
            "convblock3",
            conv_block(
                in_channels=input_channels[2],
                out_channels=output_channels[2],
                norm_layer=norm_layer,
                act_layer=act_layer,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
        )

        self.dense_layers = nn.Sequential()
        for i, (in_c, out_c) in enumerate(zip(fc_input_channels, fc_output_channels)):
            self.dense_layers.add_module(f"dense_layers_{i}", nn.Linear(in_c, out_c))
            if i != len(fc_input_channels) - 1:
                self.dense_layers.add_module("dense_layers_relu", nn.ReLU())

        self.filters_model = EnhancementFilters(parameters_foo, parameters_amounts)

    def forward(self, x):
        """Forward data to model"""

        out = F.interpolate(x, size=(self.image_size, self.image_size))
        out = self.conv_blocks(out)
        out = torch.flatten(out, 1)

        parameters = self.dense_layers(out)
        out = self.filters_model(x, parameters)

        return out
