import os
from typing import Dict
from numpy.typing import NDArray


from .model_parameters_utils import load_model_parameters


class NoImageModelSpesifiedError(Exception): pass


def get_vgg_19_model_path() -> str:
    try:
        return os.environ['AA_VGG_19']
    except KeyError as variable_not_found:
        raise NoImageModelSpesifiedError('No pretrained image model found. '
            'Please download it and set the AA_VGG_19 environment variable with the'
            'path where ou stored the model (*.mat file), to indicate to wher to '
            'locate and load it') from variable_not_found


def load_default_model_parameters() -> Dict[str, NDArray]:
    path = get_vgg_19_model_path()
    return load_model_parameters(path)
