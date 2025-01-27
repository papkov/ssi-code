from typing import Any, Dict, Tuple, Union

import numpy
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve
from torch import Tensor as T

from ssi.it_ptcnn import PTCNNImageTranslator
from ssi.models.psf_convolution import PSFConvolutionLayer
from ssi.utils.log.log import lprint


class SSIDeconvolution(PTCNNImageTranslator):
    """
    Pytorch-based CNN image deconvolution
    """

    def __init__(
        self,
        psf_kernel: numpy.ndarray = None,
        broaden_psf: int = 1,
        sharpening: float = 0.0,
        bounds_loss: float = 0.1,
        entropy: float = 0.0,
        clip_before_psf: bool = True,
        fft_psf: Union[str, bool] = "auto",
        **kwargs,
    ):
        """
        Constructs a CNN image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)

        :param clip_before_psf: torch.clamp(x, 0, 1) before PSF convolution
        :param fft_psf: "auto" or True or False
        """
        super().__init__(**kwargs)

        if self.standardize_image and clip_before_psf:
            lprint(
                "Clipping before PSF convolution is not supported when standardizing image"
            )
            clip_before_psf = False

        self.provided_psf_kernel = psf_kernel
        self.broaden_psf = broaden_psf
        self.sharpening = sharpening
        self.bounds_loss = bounds_loss
        self.entropy = entropy
        self.clip_before_psf = clip_before_psf
        self.fft_psf = fft_psf

    def _train(
        self,
        input_image,
        target_image,
        tile_size=None,
        train_valid_ratio=0.1,
        callback_period=3,
        jinv=False,
    ):

        ndim = input_image.ndim - 2
        num_channels = input_image.shape[1]

        self.psf_kernel = self.provided_psf_kernel

        for i in range(self.broaden_psf):

            self.psf_kernel = numpy.pad(
                self.psf_kernel, (1,), mode="constant", constant_values=0
            )

            broadening_kernel = None
            if ndim == 2:
                broadening_kernel = numpy.array(
                    [[0.095, 0.14, 0.095], [0.14, 0.2, 0.14], [0.095, 0.14, 0.095]]
                )
            elif ndim == 3:
                broadening_kernel = numpy.array(
                    [
                        [
                            [0.095, 0.095, 0.095],
                            [0.095, 0.140, 0.095],
                            [0.095, 0.095, 0.095],
                        ],
                        [
                            [0.095, 0.140, 0.095],
                            [0.140, 0.200, 0.140],
                            [0.095, 0.140, 0.095],
                        ],
                        [
                            [0.095, 0.095, 0.095],
                            [0.095, 0.140, 0.095],
                            [0.095, 0.095, 0.095],
                        ],
                    ]
                )

            broaden_kernel = broadening_kernel / broadening_kernel.sum()
            self.psf_kernel = convolve(
                self.psf_kernel,
                broaden_kernel,
                mode="constant",
            )

        self.psf_kernel /= self.psf_kernel.sum()
        self.psf_kernel = self.psf_kernel.astype(numpy.float32)

        self.psf_kernel = self.psf_kernel
        self.psf_kernel_tensor = torch.from_numpy(
            self.psf_kernel[numpy.newaxis, numpy.newaxis, ...]
        ).to(self.device)

        self.psf_conv = PSFConvolutionLayer(
            self.psf_kernel,
            in_channels=num_channels,
            pad_mode="reflect" if ndim == 2 else "replicate",
            fft=self.fft_psf,
        ).to(self.device)

        super()._train(
            input_image,
            target_image,
            tile_size,
            train_valid_ratio,
            callback_period,
            jinv,
        )

    def _train_loop(self, data_loader, optimizer):
        try:
            self.model.kernel_continuity_regularisation = False
        except AttributeError:
            lprint("Cannot deactivate kernel continuity regularisation")

        super()._train_loop(data_loader, optimizer)

    def _additional_losses(
        self, translated_image: T, forward_model_image: T
    ) -> Tuple[T, Dict[str, Any]]:

        loss = 0
        loss_log = {}

        # Bounds loss:
        if self.bounds_loss and self.bounds_loss != 0:
            epsilon = 0 * 1e-8
            bounds_loss = F.relu(-translated_image - epsilon)
            bounds_loss = bounds_loss + F.relu(translated_image - 1 - epsilon)
            bounds_loss = bounds_loss.mean()
            lprint(f"bounds_loss_value = {bounds_loss}")
            loss_log["bounds_loss"] = bounds_loss.item()
            loss += self.bounds_loss * bounds_loss ** 2

        # Sharpen loss_deconvolution:
        if self.sharpening and self.sharpening != 0:
            image_for_loss = translated_image
            num_elements = image_for_loss[0, 0].nelement()
            sharpening_loss = -torch.norm(
                image_for_loss, dim=(2, 3), keepdim=True, p=2
            ) / (
                num_elements ** 2
            )  # /torch.norm(image_for_loss, dim=(2, 3), keepdim=True, p=1)
            lprint(f"sharpening loss = {sharpening_loss}")
            loss_log["sharpening_loss"] = sharpening_loss.item()
            loss += self.sharpening * sharpening_loss.mean()

        # Max entropy loss:
        if self.entropy and self.entropy != 0:
            entropy_value = entropy(translated_image)
            lprint(f"entropy_value = {entropy_value}")
            loss_log["entropy_loss"] = entropy_value.item()
            loss += -self.entropy * entropy_value

        return loss, loss_log

    def _forward_model(self, x):
        if self.clip_before_psf:
            x = torch.clamp(x, 0, 1)
        return self.psf_conv(x)


def entropy(image, normalise=True, epsilon=1e-10, clip=True):
    if clip:
        image = torch.clamp(image, 0, 1)
    image = (
        image / (epsilon + torch.sum(image, dim=(2, 3), keepdim=True))
        if normalise
        else image
    )
    entropy = -torch.where(image > 0, image * (image + epsilon).log(), image.new([0.0]))
    entropy_value = entropy.sum(dim=(2, 3), keepdim=True).mean()
    return entropy_value
