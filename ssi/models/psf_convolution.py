from functools import partial
from typing import Union

import numpy as np
import torch
import torch.nn.functional as f
from numpy import ndarray
from torch import nn

from ssi.utils.fft import fft_conv


class PSFConvolutionLayer2D(nn.Module):
    def __init__(self, kernel_psf, num_channels=1):
        kernel_size = kernel_psf.shape[0]
        super().__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size,
                stride=1,
                padding=0,
                bias=False,
                groups=num_channels,
            ),
        )

        self.weights_init(kernel_psf)

    def weights_init(self, kernel_psf):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel_psf))

    def forward(self, x):
        return self.seq(x)


class PSFConvolutionLayer3D(nn.Module):
    def __init__(self, kernel_psf, num_channels=1):
        kernel_size = kernel_psf.shape[0]
        super().__init__()
        self.seq = nn.Sequential(
            nn.ReplicationPad3d((kernel_size - 1) // 2),
            nn.Conv3d(
                num_channels,
                num_channels,
                kernel_size,
                stride=1,
                padding=0,
                bias=False,
                groups=num_channels,
            ),
        )

        self.weights_init(kernel_psf)

    def weights_init(self, kernel_psf):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel_psf))

    def forward(self, x):
        return self.seq(x)


class PSFConvolutionLayer(nn.Module):
    def __init__(
        self,
        kernel_psf: ndarray,
        in_channels: int = 1,
        pad_mode: str = "reflect",
        trainable: bool = False,
        fft: Union[str, bool] = "auto",
        auto_padding: bool = False,
    ):
        """
        Parametrized trainable version of PSF
        :param kernel_psf: ndarray, PSF kernel (should be of required layer dimensionality)
        :param in_channels: number of input channels
        :param pad_mode: "reflect" for 2D, "replicate" for 3D
        :param trainable: if True, the kernel is trainable
        :param fft: ["auto", True, False] - if "auto" - use FFT if PSF has > 100 elements
        :param auto_padding: (bool) If True, automatically computes padding based on the
                             signal size, kernel size and stride.
        """
        super().__init__()
        self.kernel_size = kernel_psf.shape
        self.n_dim = len(kernel_psf.shape)
        self.in_channels = in_channels
        assert self.n_dim in (2, 3)

        if pad_mode is None:
            pad_mode = "reflect"

        if self.n_dim == 3 and pad_mode == "reflect":
            # Not supported yet
            pad_mode = "replicate"

        self.pad_mode = pad_mode
        self.pad = [k // 2 for k in self.kernel_size]

        self.fft = fft
        if self.fft == "auto":
            # Use FFT Conv if kernel has > 100 elements
            self.fft = np.product(self.kernel_size) > 100
        if isinstance(self.fft, str):
            raise ValueError(f"Invalid fft value {self.fft}")

        if not self.fft:
            auto_padding = False
        self.auto_padding = auto_padding

        self.psf = torch.from_numpy(kernel_psf.squeeze()[(None,) * 2]).float()
        self.psf = nn.Parameter(self.psf, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.fft:
            conv = partial(
                fft_conv,
                padding_mode=self.pad_mode,
                padding="same" if self.auto_padding else self.pad,
            )
        else:
            signal_padding = tuple(np.repeat(self.pad[::-1], 2))
            x = f.pad(x, signal_padding, mode=self.pad_mode)
            conv = torch.conv2d if self.n_dim == 2 else torch.conv3d

        x = conv(x, self.psf, groups=self.in_channels, stride=1)
        return x
