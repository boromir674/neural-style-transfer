# Part of this code is due to the MatConvNet team and is used to load the
#  parameters of the pretrained VGG19 model in the notebook

from typing import Dict, Tuple
from numpy.typing import NDArray
import scipy.io


def load_model_parameters(path: str) -> Dict[str, NDArray]:
    return scipy.io.loadmat(path)


def get_layers(model_parameters: Dict[str, NDArray]) -> NDArray:
    return model_parameters['layers'][0]


def vgg_weights(layer: NDArray) -> Tuple[NDArray, NDArray]:
    """Get the weight values of a convolutional layer from the vgg model.

    Gets the weights in the form of the W and b matrices (ie in equation Wx+b)

    Args:
        layer (NDArray): the convolutional layer represented as an array

    Returns:
        (Tuple[NDArray, NDArray]): the W and b matrices with weight values
    """
    return layer[0][0][2][0][0], layer[0][0][2][0][1]
