from functools import partial
from typing import Tuple

import numpy as np
from torch import Tensor as T
from torch.utils.data import Dataset

from ssi.utils.array.nd import extract_tiles


class DeconvolutionDataset(Dataset):
    def __init__(
        self,
        input_image: np.ndarray,
        target_image: np.ndarray,
        tile_size: int,
        batch_size: int,
        self_supervised: bool,
        validation_voxels: np.ndarray,
    ) -> None:
        """"""
        self.batch_size = batch_size

        if batch_size > 1:
            input_image = np.concatenate([input_image for _ in range(16)], axis=0)
            target_image = np.concatenate([target_image for _ in range(16)], axis=0)

        num_channels_input = input_image.shape[1]
        num_channels_target = target_image.shape[1]

        extract = partial(
            extract_tiles, tile_size=tile_size, extraction_step=tile_size, flatten=True
        )

        bc_flat_input_image = input_image.reshape(-1, *input_image.shape[2:])
        bc_flat_input_tiles = np.concatenate([extract(x) for x in bc_flat_input_image])
        self.input_tiles = bc_flat_input_tiles.reshape(
            -1, num_channels_input, *bc_flat_input_tiles.shape[1:]
        )

        if self_supervised:
            self.target_tiles = self.input_tiles
        else:
            bc_flat_target_image = target_image.reshape(-1, *target_image.shape[2:])
            bc_flat_target_tiles = np.concatenate(
                [extract(x) for x in bc_flat_target_image]
            )
            self.target_tiles = bc_flat_target_tiles.reshape(
                -1, num_channels_target, *bc_flat_target_tiles.shape[1:]
            )

        mask_image = np.zeros_like(input_image)
        mask_image[validation_voxels] = 1

        bc_flat_mask_image = mask_image.reshape(-1, *mask_image.shape[2:])
        bc_flat_mask_tiles = np.concatenate([extract(x) for x in bc_flat_mask_image])
        self.mask_tiles = bc_flat_mask_tiles.reshape(
            -1, num_channels_input, *bc_flat_mask_tiles.shape[1:]
        )

    def __len__(self):
        if self.batch_size > 1:
            return 1
        else:
            return len(self.input_tiles)

    def __getitem__(self, index) -> Tuple[T, T, T]:
        if self.batch_size > 1:
            inp = self.input_tiles[0, ...]
            target = self.target_tiles[0, ...]
            mask = self.mask_tiles[0, ...]
        else:
            inp = self.input_tiles[index, ...]
            target = self.target_tiles[index, ...]
            mask = self.mask_tiles[index, ...]

        return inp, target, mask
