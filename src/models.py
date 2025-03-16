"""Models definition"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class EnhancementFilters(nn.Module):
    """Filters implementation"""

    def __init__(self, parameters_foo, parameters_amounts):
        super(EnhancementFilters, self).__init__()

        self.parameters_foo = parameters_foo
        self.parameters_amounts = parameters_amounts

        # sharp
        self.universal_kernel_sharp = {}
        self.universal_kernel_sharp["K"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, -476, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_sharp["K"] = torch.stack(
            [self.universal_kernel_sharp["K"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_sharp["K"] = torch.stack(
            [self.universal_kernel_sharp["K"]] * 3, axis=0
        ).cuda()

        self.universal_kernel_sharp["M"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [0.8, 0.8, 0.8, 0.8, 0.8],
                        [0.8, 0.9, 0.9, 0.9, 0.8],
                        [0.8, 0.9, 1.0, 0.9, 0.8],
                        [0.8, 0.9, 0.9, 0.9, 0.8],
                        [0.8, 0.8, 0.8, 0.8, 0.8],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_sharp["M"] = torch.stack(
            [self.universal_kernel_sharp["M"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_sharp["M"] = torch.stack(
            [self.universal_kernel_sharp["M"]] * 3, axis=0
        ).cuda()

    def _saturation(self, x, parameters):
        """Saturation filter"""

        out = x.clone()
        m = torch.mean(x, [2, 3]).unsqueeze(2).unsqueeze(3).expand(x.shape)
        parameters = torch.nn.Tanh()(parameters.unsqueeze(2).unsqueeze(3)).expand(
            x.shape
        )

        for i in range(parameters.shape[0]):
            if parameters[i, 0, 0, 0] > 0:
                out[i] = (m[i] - x[i]) * (1 - 1 / parameters[i])
            else:
                out[i] = -(m[i] - x[i]) * parameters[i]

        return out

    def _contrast(self, x, parameters):
        """Contrast filter"""

        out = x.clone()
        parameters = torch.nn.Tanh()(parameters.unsqueeze(2).unsqueeze(3)).expand(
            x.shape
        )

        out = out - 0.5
        for i in range(out.shape[0]):
            if parameters[i, 0, 0, 0] > 0:
                out[i] = x[i] * (1 / (1 - parameters[i]))
            else:
                out[i] = x[i] * (1 - parameters[i])

        return out

    def _white_balance(self, x, parameters):
        """White balance filter"""

        parameters = parameters.unsqueeze(2).unsqueeze(3)

        return x * parameters

    def _exposure(self, x, parameters):
        """Exposure filter"""

        return x * (2 ** parameters.unsqueeze(2).unsqueeze(3))

    def _linear(self, x, parameters):
        """Linear fliter"""

        parameters = torch.nn.Tanh()(parameters)
        P = torch.reshape(parameters[:, :9], (-1, 3, 3))  # 40, 3, 3
        b = parameters[:, 9:]  # 40, 3

        P = (
            P.unsqueeze(3)
            .unsqueeze(4)
            .expand(x.shape[0], 3, 3, x.shape[2], x.shape[3])
            .permute(0, 3, 4, 1, 2)
        )
        x = x.permute(0, 2, 3, 1).unsqueeze(4)
        return torch.matmul(P, x).squeeze().permute(0, 3, 1, 2)

    def _universal_kernel(self, x, parameters):
        """Universal kernel filter"""

        res = []

        q = parameters.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        K = self.universal_kernel_sharp["K"]
        M = self.universal_kernel_sharp["M"]

        for i in range(q.shape[0]):
            kernel = K + M * q[i]

            V = torch.sum(kernel)
            kernel = kernel / V

            weight = nn.Parameter(data=kernel, requires_grad=True)
            out = F.conv2d(x[i].unsqueeze(0), weight, padding=2)
            res.append(out)

        out = torch.stack(res, axis=0).squeeze()
        return out

    def forward(self, x, parameters):
        """Forward data"""

        results = [x]
        cur_index = 0
        for f, amount in zip(self.parameters_foo, self.parameters_amounts):
            cur_parameters = parameters[:, cur_index : (cur_index + amount)]
            result = None
            if f == "white_balance":
                result = self._white_balance(x, cur_parameters)
            elif f == "exposure":
                result = self._exposure(x, cur_parameters)
            elif f == "saturation":
                result = self._saturation(x, cur_parameters)
            elif f == "contrast":
                result = self._contrast(x, cur_parameters)
            elif f == "linear":
                result = self._linear(x, cur_parameters)
            elif "universal_kernel" in f:
                result = self._universal_kernel(
                    x, cur_parameters
                )
            results.append(result)
            cur_index += amount

        return torch.clamp(sum(results), min=0, max=1)


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
