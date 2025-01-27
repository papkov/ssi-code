import os
from pathlib import Path

import hydra
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ssi.demo.demo2D import demo as demo2D
from ssi.demo.demo3D import demo as demo3D
from ssi.utils.results import fix_seed, get_benchmark_image


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"
    print(
        f"Run experiment [{cfg.experiment}] for image [{cfg.image}], work in {os.getcwd()}"
    )
    cwd = Path(get_original_cwd())

    fix_seed(cfg.seed)

    if not cfg.check:
        wandb.init(project=cfg.project, config=dict(cfg))

    if cfg.experiment.lower() == "2d":
        image, _ = get_benchmark_image(
            "gt", cfg.image, generic_2d_mono_raw_folder=cwd / cfg.data
        )
        demo2D(
            image,
            two_pass=cfg.two_pass,
            inv_mse_before_forward_model=cfg.inv_mse_before_forward_model,
            inv_mse_lambda=cfg.inv_mse_lambda,
            masking_density=cfg.masking_density,
            training_noise=cfg.training_noise,
            max_epochs=cfg.max_epochs if not cfg.check else 10,
            patience=cfg.patience,
            learning_rate=cfg.lr,
            loss=cfg.loss,
            optimizer=cfg.optimizer,
            scheduler=cfg.scheduler,
            output_dir="images",
            check=cfg.check,
            fft_psf=cfg.fft_psf,
            clip_before_psf=cfg.clip_before_psf,
            standardize=cfg.standardize,
            amp=cfg.amp,
        )
    elif cfg.experiment.lower() == "3d":
        from skimage import data

        image = data.binary_blobs(
            length=64, n_dim=3, blob_size_fraction=0.1, seed=cfg.seed
        )
        demo3D(image)
    else:
        raise ValueError(
            f"Experiment {cfg.experiment} not supported, select one from {['2d', '3d']}"
        )


if __name__ == "__main__":
    main()
