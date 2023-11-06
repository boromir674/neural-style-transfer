from typing import Dict, Tuple

import scipy.io
from numpy.typing import NDArray

from artificial_artwork.pretrained_model import Modelhandler, ModelHandlerFacility
from artificial_artwork.pretrained_model.model_routines import PretrainedModelRoutines


class VggModelRoutines(PretrainedModelRoutines):
    def load_layers(self, file_path: str) -> NDArray:
        return scipy.io.loadmat(file_path)["layers"][0]

    def get_id(self, layer: NDArray) -> str:
        return layer[0][0][0][0]

    def get_layers_dict(self, layers: NDArray) -> Dict[str, NDArray]:
        return {self.get_id(layer): layers[index] for index, layer in enumerate(layers)}

    def get_weights(self, layer: NDArray) -> Tuple[NDArray, NDArray]:
        return layer[0][0][2][0][0], layer[0][0][2][0][1]


vgg_model_routines = VggModelRoutines()


@ModelHandlerFacility.factory.register_as_subclass("vgg")
class VggModelHandler(Modelhandler):
    @property
    def environment_variable(self) -> str:
        return "AA_VGG_19"

    @property
    def model_routines(self) -> VggModelRoutines:
        return vgg_model_routines

    @property
    def model_load_exception_text(self) -> str:
        return (
            "No pretrained image model found. "
            f"Please download it and set the {self.environment_variable} "
            "environment variable with the path where you stored the model "
            "(*.mat file), to instruct the program where to locate and load it"
        )
