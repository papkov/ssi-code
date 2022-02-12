import argparse
import os
import sys
import time
from pathlib import Path

import numpy
import numpy as np
from imageio import imread, imwrite

from ssi.lr_deconv import ImageTranslatorLRDeconv
from ssi.models.unet import UNet
from ssi.ssi_deconv import SSIDeconvolution
from ssi.utils.io.datasets import (
    add_microscope_blur_2d,
    add_poisson_gaussian_noise,
    normalise,
)
from ssi.utils.metrics.image_metrics import (
    mutual_information,
    psnr,
    spectral_mutual_information,
    ssim,
)

try:
    import napari

    use_napari = True
except ImportError:
    print("napari not installed, disable visualization")
    use_napari = False


generic_2d_mono_raw_folder = Path("ssi/benchmark/images/generic_2d_all")


def get_benchmark_image(type: str, name: str):
    folder = generic_2d_mono_raw_folder / type
    if not folder.exists():
        folder = Path("../..") / folder
    try:
        files = [f for f in folder.iterdir() if f.is_file()]
    except FileNotFoundError as e:
        print("File not found, cwd:", os.getcwd())
        raise e
    filename = [f.name for f in files if name in f.name][0]
    filepath = folder / filename
    array = imread(filepath)
    return array, filename


def printscore(header, val1, val2, val3, val4):
    print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")


def demo(
    image_clipped: np.ndarray,
    two_pass: bool = False,
    inv_mse_before_forward_model: bool = False,
    learning_rate: float = 0.01,
    max_epochs: int = 3000,
    masking_density: float = 0.01,
):

    image_clipped = normalise(image_clipped.astype(numpy.float32))
    blurred_image, psf_kernel = add_microscope_blur_2d(image_clipped)
    # noisy_blurred_image = add_noise(blurred_image, intensity=None, variance=0.01, sap=0.01, clip=True)
    noisy_blurred_image = add_poisson_gaussian_noise(
        blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10
    )

    lr = ImageTranslatorLRDeconv(psf_kernel=psf_kernel, backend="cupy")
    lr.train(noisy_blurred_image)
    lr.max_num_iterations = 2
    lr_deconvolved_image_2 = lr.translate(noisy_blurred_image)
    lr.max_num_iterations = 5
    lr_deconvolved_image_5 = lr.translate(noisy_blurred_image)
    lr.max_num_iterations = 10
    lr_deconvolved_image_10 = lr.translate(noisy_blurred_image)
    lr.max_num_iterations = 20
    lr_deconvolved_image_20 = lr.translate(noisy_blurred_image)

    it_deconv = SSIDeconvolution(
        max_epochs=max_epochs,
        patience=300,
        batch_size=8,
        learning_rate=learning_rate,
        normaliser_type="identity",
        psf_kernel=psf_kernel,
        model_class=UNet,
        masking=True,
        masking_density=masking_density,
        loss=loss,
        two_pass=two_pass,
        inv_mse_before_forward_model=inv_mse_before_forward_model,
    )

    start = time.time()
    it_deconv.train(noisy_blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    start = time.time()
    deconvolved_image = it_deconv.translate(noisy_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    image_clipped = numpy.clip(image_clipped, 0, 1)
    lr_deconvolved_image_2_clipped = numpy.clip(lr_deconvolved_image_2, 0, 1)
    lr_deconvolved_image_5_clipped = numpy.clip(lr_deconvolved_image_5, 0, 1)
    lr_deconvolved_image_10_clipped = numpy.clip(lr_deconvolved_image_10, 0, 1)
    lr_deconvolved_image_20_clipped = numpy.clip(lr_deconvolved_image_20, 0, 1)
    deconvolved_image_clipped = numpy.clip(deconvolved_image, 0, 1)

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")
    printscore(
        "blurry image          :   ",
        psnr(image_clipped, blurred_image),
        spectral_mutual_information(image_clipped, blurred_image),
        mutual_information(image_clipped, blurred_image),
        ssim(image_clipped, blurred_image),
    )

    printscore(
        "noisy and blurry image:   ",
        psnr(image_clipped, noisy_blurred_image),
        spectral_mutual_information(image_clipped, noisy_blurred_image),
        mutual_information(image_clipped, noisy_blurred_image),
        ssim(image_clipped, noisy_blurred_image),
    )

    printscore(
        "lr deconv (n=2)       :    ",
        psnr(image_clipped, lr_deconvolved_image_2_clipped),
        spectral_mutual_information(image_clipped, lr_deconvolved_image_2_clipped),
        mutual_information(image_clipped, lr_deconvolved_image_2_clipped),
        ssim(image_clipped, lr_deconvolved_image_2_clipped),
    )

    printscore(
        "lr deconv (n=5)       :    ",
        psnr(image_clipped, lr_deconvolved_image_5_clipped),
        spectral_mutual_information(image_clipped, lr_deconvolved_image_5_clipped),
        mutual_information(image_clipped, lr_deconvolved_image_5_clipped),
        ssim(image_clipped, lr_deconvolved_image_5_clipped),
    )

    printscore(
        "lr deconv (n=10)      :    ",
        psnr(image_clipped, lr_deconvolved_image_10_clipped),
        spectral_mutual_information(image_clipped, lr_deconvolved_image_10_clipped),
        mutual_information(image_clipped, lr_deconvolved_image_10_clipped),
        ssim(image_clipped, lr_deconvolved_image_10_clipped),
    )

    printscore(
        "lr deconv (n=20)      :    ",
        psnr(image_clipped, lr_deconvolved_image_20_clipped),
        spectral_mutual_information(image_clipped, lr_deconvolved_image_20_clipped),
        mutual_information(image_clipped, lr_deconvolved_image_20_clipped),
        ssim(image_clipped, lr_deconvolved_image_20_clipped),
    )

    printscore(
        "ssi deconv            : ",
        psnr(image_clipped, deconvolved_image_clipped),
        spectral_mutual_information(image_clipped, deconvolved_image_clipped),
        mutual_information(image_clipped, deconvolved_image_clipped),
        ssim(image_clipped, deconvolved_image_clipped),
    )

    print(
        "NOTE: if you get a bad results for ssi, blame stochastic optimisation and retry..."
    )
    print(
        "      The training is done on the same exact image that we infer on, very few pixels..."
    )
    print("      Training should be more stable given more data...")

    if use_napari:
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name="image")
            viewer.add_image(blurred_image, name="blurred")
            viewer.add_image(noisy_blurred_image, name="noisy_blurred_image")
            viewer.add_image(
                lr_deconvolved_image_2_clipped, name="lr_deconvolved_image_2"
            )
            viewer.add_image(
                lr_deconvolved_image_5_clipped, name="lr_deconvolved_image_5"
            )
            viewer.add_image(
                lr_deconvolved_image_10_clipped, name="lr_deconvolved_image_10"
            )
            viewer.add_image(
                lr_deconvolved_image_20_clipped, name="lr_deconvolved_image_20"
            )
            viewer.add_image(deconvolved_image_clipped, name="ssi_deconvolved_image")
    else:
        imwrite("demo_results/image.png", image, format="png")
        imwrite("demo_results/blurred.png", blurred_image, format="png")
        imwrite(
            "demo_results/noisy_blurred_image.png", noisy_blurred_image, format="png"
        )
        imwrite(
            "demo_results/lr_deconvolved_image_2.png",
            lr_deconvolved_image_2_clipped,
            format="png",
        )
        imwrite(
            "demo_results/lr_deconvolved_image_5.png",
            lr_deconvolved_image_5_clipped,
            format="png",
        )
        imwrite(
            "demo_results/lr_deconvolved_image_10.png",
            lr_deconvolved_image_10_clipped,
            format="png",
        )
        imwrite(
            "demo_results/lr_deconvolved_image_20.png",
            lr_deconvolved_image_20_clipped,
            format="png",
        )
        imwrite(
            "demo_results/ssi_deconvolved_image.png",
            deconvolved_image_clipped,
            format="png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSI Demo 2D")
    parser.add_argument(
        "--image", "-i", type=str, default="drosophila", help="Image to test on"
    )
    parser.add_argument("--masking_density", "-m", type=float, default=0.01)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.01)
    parser.add_argument("--max_epochs", "-e", type=int, default=3000)
    parser.add_argument(
        "--two_pass",
        "-t",
        action="store_true",
        default=False,
        help="Use two-pass scheme from Noise2Same",
    )
    parser.add_argument(
        "--inv_mse_before_forward_model",
        "-imb",
        action="store_true",
        default=False,
        help="Calculate inverse mse before forward PSF model",
    )

    args = parser.parse_args()
    image, _ = get_benchmark_image("gt", args.image)
    postfix = f"two_pass={args.two_pass}_before={args.inv_mse_before_forward_model}"
    demo(
        image,
        two_pass=args.two_pass,
        inv_mse_before_forward_model=args.inv_mse_before_forward_model,
        masking_density=args.masking_density,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )
