import math
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Dict, Tuple

import numpy
import torch
import wandb
from torch import Tensor as T
from torch import nn
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from ssi.base import ImageTranslatorBase
from ssi.models.masking import Masking
from ssi.models.unet import UNet
from ssi.optimisers.esadam import ESAdam
from ssi.utils.data.dataset import DeconvolutionDataset
from ssi.utils.log.log import lprint, lsection


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


class PTCNNImageTranslator(ImageTranslatorBase):
    """
    Pytorch-based CNN image translator
    """

    def __init__(
        self,
        max_epochs=2048,
        patience=None,
        patience_epsilon=0.0,
        learning_rate=0.01,
        batch_size=8,
        model_class=UNet,
        masking=True,
        two_pass=False,  # two-pass Noise2Same loss
        inv_mse_lambda: float = 2.0,
        inv_mse_before_forward_model=False,
        masking_density=0.01,
        loss="l1",
        normaliser_type="percentile",
        balance_training_data=None,
        keep_ratio=1,
        max_voxels_for_training=4e6,
        monitor=None,
        use_cuda=True,
        device_index=0,
        max_tile_size: int = 1024,  # TODO: adjust based on available memory
        check: bool = True,
        optimizer: str = "esadam",
        standardize: bool = False,
        amp: bool = False,
    ):
        """
        Constructs an image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        :param two_pass: bool, adopt Noise2Same two forward pass strategy (one masked, one unmasked)
        :param inv_mse_before_forward_model: bool, use invariance MSE before forward (PSF) model for Noise2Same
        :param check: bool, run smoke test
        :param optimizer: str, optimiser to use ["adam", "esadam"]
        :param standardize: bool, standardize input images to zero mean and unit variance
        """
        super().__init__(normaliser_type, monitor=monitor)
        if two_pass and not masking:
            lprint("Force masking=True, it is needed in two-pass")
            masking = True

        use_cuda = use_cuda and (torch.cuda.device_count() > 0)
        self.device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
        lprint(f"Using device: {self.device}")

        self.max_epochs = max_epochs
        self.patience = max_epochs if patience is None else patience
        self.patience_epsilon = patience_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss = loss
        self.max_voxels_for_training = max_voxels_for_training
        self.keep_ratio = keep_ratio
        self.balance_training_data = balance_training_data

        self.model_class = model_class

        self.l1_weight_regularisation = 1e-6
        self.l2_weight_regularisation = 1e-6
        self.training_noise = 0.1
        self.reload_best_model_period = max_epochs  # //2
        self.reduce_lr_patience = patience // 2
        self.reduce_lr_factor = 0.9
        self.masking = masking
        self.two_pass = two_pass
        self.inv_mse_before_forward_model = inv_mse_before_forward_model
        self.inv_mse_lambda = inv_mse_lambda
        self.masking_density = masking_density
        self.optimizer_class = ESAdam if optimizer == "esadam" else torch.optim.Adam
        self.max_tile_size = max_tile_size

        self._stop_training_flag = False
        self.check = check
        self.standardize = standardize
        self.amp = amp

        # Denoise loss function:
        loss_function = nn.L1Loss()
        if self.loss.lower() == "l2":
            lprint(f"Training/Validation loss: L2")
            if self.masking:
                loss_function = (
                    lambda u, v, m: (u - v) ** 2 if m is None else ((u - v) * m) ** 2
                )
            else:
                loss_function = lambda u, v: (u - v) ** 2

        elif self.loss.lower() == "l1":
            lprint(f"Training/Validation loss: L1")
            if self.masking:
                loss_function = (
                    lambda u, v, m: torch.abs(u - v)
                    if m is None
                    else torch.abs((u - v) * m)
                )
            else:
                loss_function = lambda u, v: torch.abs(u - v)
            lprint(f"Training/Validation loss: L1")

        self.loss_function = loss_function

        # Monitor
        self.best_val_loss_value = math.inf
        self.best_model_state_dict = None
        self.patience_counter = 0

    def _train(
        self,
        input_image,
        target_image,
        tile_size=None,
        train_valid_ratio=0.1,
        callback_period=3,
        j_inv=False,
    ):
        self._stop_training_flag = False

        if j_inv is not None and not j_inv:
            self.masking = False

        shape = input_image.shape
        num_input_channels = input_image.shape[1]
        num_output_channels = target_image.shape[1]
        num_spatiotemp_dim = input_image.ndim - 2

        # tile size:
        if tile_size is None:
            # tile_size = min(self.max_tile_size, min(shape[2:]))
            tile_size = tuple(min(self.max_tile_size, s) for s in shape[2:])
            lprint(f"Estimated max tile size {tile_size}")

        # Decide on how many voxels to be used for validation:
        num_val_voxels = int(train_valid_ratio * input_image.size)
        lprint(
            f"Number of voxels used for validation: {num_val_voxels} (train_valid_ratio={train_valid_ratio})"
        )

        # Generate random coordinates for these voxels:
        val_voxels = tuple(numpy.random.randint(d, size=num_val_voxels) for d in shape)
        lprint(f"Validation voxel coordinates: {val_voxels}")

        # Training Tile size:
        lprint(f"Train Tile dimensions: {tile_size}")

        # Prepare Training Dataset:
        dataset = self._get_dataset(
            input_image,
            target_image,
            self.self_supervised,
            tile_size=tile_size,
            mode="grid",
            validation_voxels=val_voxels,
            batch_size=self.batch_size,
        )
        lprint(f"Number tiles for training: {len(dataset)}")

        # Training Data Loader:
        # num_workers = max(3, os.cpu_count() // 2)
        num_workers = 0  # faster if data is already in memory...
        lprint(f"Number of workers for loading training/validation data: {num_workers}")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Model
        self.model = self.model_class(
            num_input_channels, num_output_channels, ndim=num_spatiotemp_dim
        ).to(self.device)

        number_of_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        lprint(
            f"Number of trainable parameters in {self.model_class} model: {number_of_parameters}"
        )

        if self.masking:
            self.masked_model = Masking(self.model, density=0.5).to(self.device)

        lprint(f"Optimiser class: {self.optimizer_class}")
        lprint(f"Learning rate : {self.learning_rate}")

        # Optimiser:
        if isinstance(self.optimizer_class, ESAdam):
            optimizer = partial(
                self.optimizer_class, start_noise_level=self.training_noise
            )
        else:
            optimizer = self.optimizer_class

        optimizer = optimizer(
            chain(self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.l2_weight_regularisation,
        )

        lprint(f"Optimiser: {optimizer}")

        # Start training:
        self._train_loop(data_loader, optimizer)

    def _get_dataset(
        self,
        input_image: numpy.ndarray,
        target_image: numpy.ndarray,
        self_supervised: bool,
        tile_size: int,
        mode: str,
        validation_voxels,
        batch_size=32,
    ):
        if mode == "grid":
            return DeconvolutionDataset(
                input_image=input_image,
                target_image=target_image,
                tile_size=tile_size,
                self_supervised=self_supervised,
                batch_size=batch_size,
                validation_voxels=validation_voxels,
            )
        else:
            return None

    def _train_step(
        self,
        input_images: T,
        target_images: T,
        valid_mask_images: T,
        epoch: int = 0,
    ) -> Tuple[T, Dict[str, float]]:
        self.model.train()
        loss_log = {}

        # Adding training noise to input:
        if self.training_noise > 0:
            with torch.no_grad():
                alpha = self.training_noise / (1 + (10000 * epoch / self.max_epochs))
                lprint(f"Training noise level: {alpha}")
                loss_log["training_noise"] = alpha
                training_noise = alpha * torch.randn_like(
                    input_images, device=input_images.device
                )
                input_images += training_noise

        # Forward pass:
        if self.masking:
            translated_images = self.masked_model(input_images)  # pass with masking
        else:
            translated_images = self.model(input_images)

        # apply forward model:
        forward_model_images = self._forward_model(translated_images)

        if self.two_pass:
            # pass without masking
            translated_images_full = self.model(input_images)
            forward_model_images_full = self._forward_model(translated_images_full)

            # no masking for reconstruction
            reconstruction_loss = self.loss_function(
                forward_model_images_full, target_images, None
            ).mean()
            loss_log["reconstruction_loss"] = reconstruction_loss.item()

            if self.inv_mse_before_forward_model:
                u = translated_images_full * (1 - valid_mask_images)
                v = translated_images * (1 - valid_mask_images)
            else:
                u = forward_model_images_full * (1 - valid_mask_images)
                v = forward_model_images * (1 - valid_mask_images)

            mask = self.masked_model.get_mask()
            invariance_loss = self.loss_function(u, v, mask).mean()
            loss_log["invariance_loss"] = invariance_loss.item()

            translation_loss_value = (
                reconstruction_loss + self.inv_mse_lambda * torch.sqrt(invariance_loss)
            )
        else:
            # validation masking:
            u = forward_model_images * (1 - valid_mask_images)
            v = target_images * (1 - valid_mask_images)

            # translation loss (per voxel):
            if self.masking:
                mask = self.masked_model.get_mask()
                translation_loss = self.loss_function(u, v, mask)
            else:
                translation_loss = self.loss_function(u, v)

            # translation loss all voxels
            translation_loss_value = translation_loss.mean()

        loss_log["translation_loss"] = translation_loss_value.item()

        # Additional losses:
        (
            additional_loss_value,
            additional_loss_log,
        ) = self._additional_losses(translated_images, forward_model_images)
        if additional_loss_value is not None:
            translation_loss_value += additional_loss_value
            loss_log.update(additional_loss_log)

        return translation_loss_value, loss_log

    @torch.no_grad()
    def _valid_step(self, input_images, target_images, validation_mask_images) -> float:
        self.model.eval()
        # Forward pass:
        if self.masking:
            translated_images = self.masked_model(input_images)
        else:
            translated_images = self.model(input_images)

        # apply forward model:
        forward_model_images = self._forward_model(translated_images)

        # validation masking:
        u = forward_model_images * validation_mask_images
        v = target_images * validation_mask_images

        # translation loss (per voxel):
        if self.masking:
            translation_loss = self.loss_function(u, v, None)
        else:
            translation_loss = self.loss_function(u, v)

        # loss values:
        translation_loss_value = translation_loss.mean().cpu().item()

        return translation_loss_value

    def _epoch(
        self, optimizer, data_loader, epoch: int = 0
    ) -> Tuple[float, float, Dict[str, float]]:
        train_loss_value = 0
        val_loss_value = 0
        loss_log_epoch = {}

        if hasattr(self, "masked_model"):
            self.masked_model.density = (
                0.005 * self.masking_density + 0.995 * self.masked_model.density
            )
            lprint(f"masking density: {self.masked_model.density}")

        for i, (input_images, target_images, val_mask_images) in enumerate(data_loader):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            input_images_gpu = input_images.to(self.device, non_blocking=True)
            target_images_gpu = target_images.to(self.device, non_blocking=True)
            validation_mask_images_gpu = val_mask_images.to(
                self.device, non_blocking=True
            )

            # Training step
            with autocast(enabled=self.amp):
                translation_loss_value, loss_log = self._train_step(
                    input_images_gpu,
                    target_images_gpu,
                    validation_mask_images_gpu,
                    epoch,
                )

            # Backpropagation
            translation_loss_value.backward()

            # Updating parameters
            optimizer.step()

            # post optimisation -- if needed:
            self.model.post_optimisation()

            # update training loss_deconvolution for whole image:
            train_loss_value += translation_loss_value.item()

            # Validation:
            with autocast(enabled=self.amp):
                translation_loss_value = self._valid_step(
                    input_images_gpu,
                    target_images_gpu,
                    validation_mask_images_gpu,
                )
            # update validation loss_deconvolution for whole image:
            loss_log["val_translation_loss"] = translation_loss_value
            val_loss_value += translation_loss_value

            if not loss_log_epoch:
                loss_log_epoch = deepcopy(loss_log)
            else:
                loss_log_epoch = {k: v + loss_log[k] for k, v in loss_log_epoch.items()}

        # Aggregate losses:
        iteration = len(data_loader)
        train_loss_value /= iteration
        lprint(f"Training loss value: {train_loss_value}")

        val_loss_value /= iteration
        lprint(f"Validation loss value: {val_loss_value}")

        loss_log_epoch = {k: v / iteration for k, v in loss_log_epoch.items()}

        return train_loss_value, val_loss_value, loss_log_epoch

    def _train_loop(self, data_loader, optimizer):

        # Scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.reduce_lr_factor,
            verbose=True,
            patience=self.reduce_lr_patience,
        )

        self.best_val_loss_value = math.inf
        self.best_model_state_dict = None
        self.patience_counter = 0

        with lsection(f"Training loop:"):
            lprint(f"Maximum number of epochs: {self.max_epochs}")
            lprint(
                f"Training type: {'self-supervised' if self.self_supervised else 'supervised'}"
            )

            for epoch in range(self.max_epochs):
                with lsection(f"Epoch {epoch}:"):
                    # One epoch of training
                    train_loss_value, val_loss_value, loss_log_epoch = self._epoch(
                        optimizer, data_loader, epoch
                    )

                    # Learning rate schedule:
                    scheduler.step(val_loss_value)

                    # Logging:
                    loss_log_epoch["masking_density"] = self.masked_model.density
                    loss_log_epoch["lr"] = scheduler._last_lr[0]

                    if not self.check:
                        wandb.log(loss_log_epoch)

                    # Monitoring and saving:
                    if val_loss_value < self.best_val_loss_value:
                        lprint(f"## New best val loss!")
                        if (
                            val_loss_value
                            < self.best_val_loss_value - self.patience_epsilon
                        ):
                            lprint(f"## Good enough to reset patience!")
                            self.patience_counter = 0

                        # Update best val loss value:
                        self.best_val_loss_value = val_loss_value

                        # Save model:
                        self.best_model_state_dict = OrderedDict(
                            {k: v.to("cpu") for k, v in self.model.state_dict().items()}
                        )

                    else:
                        if (
                            epoch % max(1, self.reload_best_model_period) == 0
                            and self.best_model_state_dict
                        ):
                            lprint(f"Reloading best models to date!")
                            self.model.load_state_dict(self.best_model_state_dict)

                        if self.patience_counter > self.patience:
                            lprint(f"Early stopping!")
                            break

                        # No improvement:
                        lprint(
                            f"No improvement of validation losses, patience = {self.patience_counter}/{self.patience} "
                        )
                        self.patience_counter += 1

                    lprint(f"## Best val loss: {self.best_val_loss_value}")

                    if self._stop_training_flag:
                        lprint(f"Training interupted!")
                        break

        lprint(f"Reloading best models to date!")
        self.model.load_state_dict(self.best_model_state_dict)

    def _additional_losses(self, translated_image, forward_model_image):
        return None, {}

    def _forward_model(self, x):
        return x

    @torch.no_grad()
    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """
        Internal method that translates an input image on the basis of the trained model.
        :param input_image: input image
        :return:
        """
        self.model.eval()
        input_image = torch.Tensor(input_image)
        input_image = input_image.to(self.device)
        inferred_image: torch.Tensor = self.model(input_image)
        inferred_image = inferred_image.detach().cpu().numpy()
        return inferred_image

    def visualise_weights(self):
        try:
            self.model.visualise_weights()
        except AttributeError:
            lprint(
                f"Method 'visualise_weights()' unavailable, cannot visualise weights. "
            )
