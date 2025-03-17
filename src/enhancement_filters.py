"""Enhancement Filters"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


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

        # soft frame
        self.universal_kernel_soft_frame = {}
        self.universal_kernel_soft_frame["K"] = (
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
        self.universal_kernel_soft_frame["K"] = torch.stack(
            [self.universal_kernel_soft_frame["K"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_soft_frame["K"] = torch.stack(
            [self.universal_kernel_soft_frame["K"]] * 3, axis=0
        ).cuda()

        self.universal_kernel_soft_frame["M"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.7, 0.7, 0.7, 0.5],
                        [0.5, 0.7, 0.5, 0.7, 0.5],
                        [0.5, 0.7, 0.7, 0.7, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_soft_frame["M"] = torch.stack(
            [self.universal_kernel_soft_frame["M"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_soft_frame["M"] = torch.stack(
            [self.universal_kernel_soft_frame["M"]] * 3, axis=0
        ).cuda()

        # universal frame
        self.universal_kernel_universal_frame = {}
        self.universal_kernel_universal_frame["K"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_universal_frame["K"] = torch.stack(
            [self.universal_kernel_universal_frame["K"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_universal_frame["K"] = torch.stack(
            [self.universal_kernel_universal_frame["K"]] * 3, axis=0
        ).cuda()

        self.universal_kernel_universal_frame["M"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.7, 0.7, 0.7, 0.5],
                        [0.5, 0.7, 0.5, 0.7, 0.5],
                        [0.5, 0.7, 0.7, 0.7, 0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_universal_frame["M"] = torch.stack(
            [self.universal_kernel_universal_frame["M"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_universal_frame["M"] = torch.stack(
            [self.universal_kernel_universal_frame["M"]] * 3, axis=0
        ).cuda()

        # universal soft-unsharp
        self.universal_kernel_universal_soft_unsharp = {}
        self.universal_kernel_universal_soft_unsharp["K"] = (
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
        self.universal_kernel_universal_soft_unsharp["K"] = torch.stack(
            [self.universal_kernel_universal_soft_unsharp["K"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_universal_soft_unsharp["K"] = torch.stack(
            [self.universal_kernel_universal_soft_unsharp["K"]] * 3, axis=0
        ).cuda()

        self.universal_kernel_universal_soft_unsharp["M"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.9, 0.9, 0.9, 1.0],
                        [1.0, 0.9, 0.8, 0.9, 1.0],
                        [1.0, 0.9, 0.9, 0.9, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_universal_soft_unsharp["M"] = torch.stack(
            [self.universal_kernel_universal_soft_unsharp["M"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_universal_soft_unsharp["M"] = torch.stack(
            [self.universal_kernel_universal_soft_unsharp["M"]] * 3, axis=0
        ).cuda()

        # universal unsharp
        self.universal_kernel_universal_unsharp = {}
        self.universal_kernel_universal_unsharp["K"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_universal_unsharp["K"] = torch.stack(
            [self.universal_kernel_universal_unsharp["K"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_universal_unsharp["K"] = torch.stack(
            [self.universal_kernel_universal_unsharp["K"]] * 3, axis=0
        ).cuda()

        self.universal_kernel_universal_unsharp["M"] = (
            torch.FloatTensor(
                np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.9, 0.9, 0.9, 1.0],
                        [1.0, 0.9, 0.8, 0.9, 1.0],
                        [1.0, 0.9, 0.9, 0.9, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.universal_kernel_universal_unsharp["M"] = torch.stack(
            [self.universal_kernel_universal_unsharp["M"]] * 3, axis=1
        ).squeeze()
        self.universal_kernel_universal_unsharp["M"] = torch.stack(
            [self.universal_kernel_universal_unsharp["M"]] * 3, axis=0
        ).cuda()

    def _gaussian_blur(self, img, sigma):
        """Gaussian Blur"""

        return torchvision.transforms.functional.gaussian_blur(
            img, kernel_size=5, sigma=sigma
        )

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

    def _channel_wise_color(self, x, parameters):
        """Channel-wise color filter"""

        out = x.clone()
        parameters = torch.nn.Tanh()(parameters.unsqueeze(2).unsqueeze(3))
        n = parameters.size()[1] // 3

        for i in range(n):
            slag = torch.abs((n - 1) * x - i + 1)
            fi = torch.max(torch.zeros_like(slag), torch.ones_like(slag) - slag)
            out = out + parameters[:, i * 3 : (i * 3 + 3)] * fi
        return out

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

    def _universal_kernel(self, x, parameters, kernel_name="sharp"):
        """Universal kernel filter"""

        res = []
        if kernel_name == "full":
            P = (
                parameters.unsqueeze(2)
                .expand((x.shape[0], 25, 3))
                .unsqueeze(3)
                .expand((x.shape[0], 25, 3, 3))
            )
            P = torch.permute(P, (0, 2, 3, 1))
            P = torch.reshape(P, (-1, 3, 3, 5, 5))

            for i in range(x.shape[0]):
                kernel = P[i]

                V = torch.sum(kernel)
                kernel = kernel / V

                weight = nn.Parameter(data=kernel, requires_grad=True)
                out = F.conv2d(x[i].unsqueeze(0), weight, padding=2)
                res.append(out)
        else:
            q = parameters.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            if kernel_name == "sharp":
                K = self.universal_kernel_sharp["K"]
                M = self.universal_kernel_sharp["M"]
            elif kernel_name == "soft_frame":
                K = self.universal_kernel_soft_frame["K"]
                M = self.universal_kernel_soft_frame["M"]
            elif kernel_name == "universal_frame":
                K = self.universal_kernel_universal_frame["K"]
                M = self.universal_kernel_universal_frame["M"]
            elif kernel_name == "universal_soft_unsharp":
                K = self.universal_kernel_universal_soft_unsharp["K"]
                M = self.universal_kernel_universal_soft_unsharp["M"]
            elif kernel_name == "universal_unsharp":
                K = self.universal_kernel_universal_unsharp["K"]
                M = self.universal_kernel_universal_unsharp["M"]

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
            elif f == 'channel_wise_color':
                result = self._channel_wise_color(x, cur_parameters)
            elif f == "exposure":
                result = self._exposure(x, cur_parameters)
            elif f == "saturation":
                result = self._saturation(x, cur_parameters)
            elif f == "contrast":
                result = self._contrast(x, cur_parameters)
            elif f == "linear":
                result = self._linear(x, cur_parameters)
            elif "universal_kernel" in f:
                result = self._universal_kernel(x, cur_parameters)
            results.append(result)
            cur_index += amount

        return torch.clamp(sum(results), min=0, max=1)
