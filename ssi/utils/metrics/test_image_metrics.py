import numpy
import pytest

from ssi.utils.io.datasets import add_noise, camera, normalise
from ssi.utils.metrics.image_metrics import (
    joint_information,
    mutual_information,
    spectral_mutual_information,
    spectral_psnr,
)


def test_spectral_psnr():
    camera_image = normalise(camera()).astype(numpy.float)
    camera_image_with_noise_high = add_noise(camera())
    camera_image_with_noise_low = add_noise(
        camera(), intensity=1000, variance=0.0001, sap=0.000001
    )

    ji_high = spectral_psnr(camera_image, camera_image_with_noise_high)
    ji_low = spectral_psnr(camera_image, camera_image_with_noise_low)

    assert ji_high > ji_low


def test_mutual_information():
    camera_image = camera()
    camera_image_with_noise = add_noise(camera())

    mi = mutual_information(camera_image, camera_image, normalised=False)
    mi_n = mutual_information(camera_image, camera_image_with_noise, normalised=False)

    assert mi > mi_n


def test_normalised_mutual_information():
    camera_image = camera()
    camera_image_with_noise = add_noise(camera())

    assert pytest.approx(
        mutual_information(camera_image, camera_image, normalised=True), 1
    )
    assert pytest.approx(
        mutual_information(
            camera_image_with_noise, camera_image_with_noise, normalised=True
        ),
        1,
    )

    assert (
        mutual_information(camera_image, camera_image_with_noise, normalised=True) < 1
    )


def test_spectral_mutual_information():
    camera_image = camera()
    camera_image_with_noise = add_noise(camera())

    smi = spectral_mutual_information(camera_image, camera_image)
    smi_n = spectral_mutual_information(camera_image, camera_image_with_noise)

    assert smi_n < smi
