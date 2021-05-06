from tv_restoration.chambole_pock import chambolle_pock, power_method
from tv_restoration.convo_operators import ConvolutionOperator, gaussian1D
from utils.io.datasets import add_noise, camera, normalise


def original_demo():
    beta = 1e-7  # weight of TV regularization
    n_it = 500  # number of iterations

    # Init.
    # ------
    image = normalise(camera().astype("f"))

    kern = gaussian1D(2.6)
    K = ConvolutionOperator(kern)
    P = lambda x: K * x
    PT = lambda x: K.T() * x

    # Run
    blurred_image = add_noise(P(image), intensity=None, variance=0.00)
    L = power_method(P, PT, blurred_image, n_it=200)
    print("||K|| = %f" % L)
    en, deconvolved_image = chambolle_pock(P, PT, blurred_image, beta, L, n_it)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(en)
    plt.show()

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name="image")
        viewer.add_image(blurred_image, name="blurred")
        viewer.add_image(deconvolved_image, name="deconvolved_image")


original_demo()
