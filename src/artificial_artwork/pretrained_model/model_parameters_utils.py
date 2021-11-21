### Part of this code is due to the MatConvNet team and is used to load the parameters of the pretrained VGG19 model in the notebook ###

from typing import Dict, Tuple
from numpy.typing import NDArray
import scipy.io


def load_model_parameters(path: str) -> Dict[str, NDArray]:
    return scipy.io.loadmat(path)


def get_layers(model_parameters: Dict[str, NDArray]) -> NDArray:
    return model_parameters['layers'][0]


def vgg_weights(layer: NDArray) -> Tuple[NDArray, NDArray]:
    """Get the weights and bias for a given layer of the VGG model."""
    # wb = vgg_layers[0][layer][0][0][2]
    wb = layer[0][0][2]
    W = wb[0][0]
    b = wb[0][1]
    return W, b
