"""Utils module"""

import os
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """Set seed for all random functions"""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ConvNormActBlock(nn.Module):
    """Building block for image enhancement module"""

    def __init__(self, in_channels, out_channels, norm_layer, act_layer, kernel_size=3, stride=2, 
                 padding=0):
        super(ConvNormActBlock, self).__init__()

        self.conv_norm_act = nn.Sequential()
        self.conv_norm_act.add_module('conv2d', nn.Conv2d(in_channels, out_channels,
                                                          kernel_size=kernel_size, stride=stride,
                                                          padding=padding))

        if norm_layer is not None:
            self.conv_norm_act.add_module('norm', norm_layer(out_channels))
        if act_layer == 'lrelu':
            self.conv_norm_act.add_module('act', nn.LeakyReLU(negative_slope=0.2))
        elif act_layer == 'relu':
            self.conv_norm_act.add_module('act', nn.ReLU())

    def forward(self, x):
        """Forward data to the block"""

        return self.conv_norm_act(x)
