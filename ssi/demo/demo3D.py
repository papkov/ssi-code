import time

import numpy

from ssi.lr_deconv import ImageTranslatorLRDeconv
from ssi.models.unet import UNet
from ssi.ssi_deconv import SSIDeconvolution
from ssi.utils.io.datasets import (
    add_microscope_blur_3d,
    add_poisson_gaussian_noise,
    normalise,
)
from ssi.utils.metrics.image_metrics import (
    mutual_information,
    psnr,
    spectral_mutual_information,
    ssim,
)
from ssi.utils.results import print_header, print_score

try:
    import napari

    use_napari = True
except ImportError:
    print("napari not installed, disable visualization")
    use_napari = False


def demo(image_clipped):
    image_clipped = normalise(image_clipped.astype(numpy.float32))
    blurred_image, psf_kernel = add_microscope_blur_3d(image_clipped)
    noisy_blurred_image = add_poisson_gaussian_noise(
        blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10
    )

    lr = ImageTranslatorLRDeconv(psf_kernel=psf_kernel, backend="cupy")
    lr.train(noisy_blurred_image)
    # lr.max_num_iterations=2
    # lr_deconvolved_image_2 = lr.translate(noisy_blurred_image)
    lr.max_num_iterations = 5
    lr_deconvolved_image_5 = lr.translate(noisy_blurred_image)
    # lr.max_num_iterations=10
    # lr_deconvolved_image_10 = lr.translate(noisy_blurred_image)
    # lr.max_num_iterations=20
    # lr_deconvolved_image_20 = lr.translate(noisy_blurred_image)

    it_deconv = SSIDeconvolution(
        max_epochs=3000,
        patience=300,
        batch_size=8,
        learning_rate=0.01,
        normaliser_type="identity",
        psf_kernel=psf_kernel,
        model_class=UNet,
        masking=True,
        masking_density=0.01,
        loss="l2",
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
    # lr_deconvolved_image_2_clipped = numpy.clip(lr_deconvolved_image_2, 0, 1)
    lr_deconvolved_image_5_clipped = numpy.clip(lr_deconvolved_image_5, 0, 1)
    # lr_deconvolved_image_10_clipped = numpy.clip(lr_deconvolved_image_10, 0, 1)
    # lr_deconvolved_image_20_clipped = numpy.clip(lr_deconvolved_image_20, 0, 1)
    deconvolved_image_clipped = numpy.clip(deconvolved_image, 0, 1)

    columns = ["PSNR", "norm spectral mutual info", "norm mutual info", "SSIM"]
    print_header(columns)
    print_score(
        "blurry image",
        psnr(image_clipped, blurred_image),
        spectral_mutual_information(image_clipped, blurred_image),
        mutual_information(image_clipped, blurred_image),
        ssim(image_clipped, blurred_image),
    )

    print_score(
        "noisy and blurry image",
        psnr(image_clipped, noisy_blurred_image),
        spectral_mutual_information(image_clipped, noisy_blurred_image),
        mutual_information(image_clipped, noisy_blurred_image),
        ssim(image_clipped, noisy_blurred_image),
    )

    # print_score(
    #     "lr deconv (n=2)",
    #     psnr(image_clipped, lr_deconvolved_image_2_clipped),
    #     spectral_mutual_information(image_clipped, lr_deconvolved_image_2_clipped),
    #     mutual_information(image_clipped, lr_deconvolved_image_2_clipped),
    #     ssim(image_clipped, lr_deconvolved_image_2_clipped),
    # )

    print_score(
        "lr deconv (n=5)",
        psnr(image_clipped, lr_deconvolved_image_5_clipped),
        spectral_mutual_information(image_clipped, lr_deconvolved_image_5_clipped),
        mutual_information(image_clipped, lr_deconvolved_image_5_clipped),
        ssim(image_clipped, lr_deconvolved_image_5_clipped),
    )

    # print_score(
    #     "lr deconv (n=10)",
    #     psnr(image_clipped, lr_deconvolved_image_10_clipped),
    #     spectral_mutual_information(image_clipped, lr_deconvolved_image_10_clipped),
    #     mutual_information(image_clipped, lr_deconvolved_image_10_clipped),
    #     ssim(image_clipped, lr_deconvolved_image_10_clipped),
    # )
    #
    # print_score(
    #     "lr deconv (n=20)",
    #     psnr(image_clipped, lr_deconvolved_image_20_clipped),
    #     spectral_mutual_information(image_clipped, lr_deconvolved_image_20_clipped),
    #     mutual_information(image_clipped, lr_deconvolved_image_20_clipped),
    #     ssim(image_clipped, lr_deconvolved_image_20_clipped),
    # )

    print_score(
        "ssi deconv",
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
            viewer.add_image(image_clipped, name="image")
            viewer.add_image(blurred_image, name="blurred")
            viewer.add_image(noisy_blurred_image, name="noisy_blurred_image")
            # viewer.add_image(lr_deconvolved_image_2_clipped, name='lr_deconvolved_image_2')
            viewer.add_image(
                lr_deconvolved_image_5_clipped, name="lr_deconvolved_image_5"
            )
            # viewer.add_image(lr_deconvolved_image_10_clipped, name='lr_deconvolved_image_10')
            # viewer.add_image(lr_deconvolved_image_20_clipped, name='lr_deconvolved_image_20')
            viewer.add_image(deconvolved_image_clipped, name="ssi_deconvolved_image")
