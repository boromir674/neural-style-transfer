"""This modules defines the interface which must be implemented in order to
utilize a pretrained model and its weights"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from numpy.typing import NDArray


class PretrainedModelRoutines(ABC):
    """Set of routines that are required in order to use a pretrained model for nst."""

    @abstractmethod
    def load_layers(self, file_path: str) -> NDArray:
        """Load a pretrained model from disk.

        Loads the model parameters, given the path to a file in the disk, that
        indicated where the pretrained model is.

        Args:
            file_path (str): the path corresponding to a file in the disk

        Returns:
            NDArray: the model parameters as a numpy array
        """
        raise NotImplementedError

    @abstractmethod
    def get_id(self, layer: NDArray) -> str:
        """Get the id of a model's network layer.

        The pretrained model being a neural network has a specific architecture
        and each layer should a unique string id that one can reference it.

        Args:
            layer (NDArray): the layer of a pretrained neural network model

        Returns:
            str: the layer id
        """
        raise NotImplementedError

    @abstractmethod
    def get_layers_dict(self, layers: NDArray) -> Dict[str, NDArray]:
        """Get a dict mapping strings to pretrained model layers.

        Args:
            layers (NDArray): the pretrained model layers

        Returns:
            Dict[str, NDArray]: the dictionary mapping strings to layers
        """
        raise NotImplementedError

    @abstractmethod
    def get_weights(self, layer: NDArray) -> Tuple[NDArray, NDArray]:
        """Get the values of the weights of a given network layer.

        Each pretrained model network layer has "learned" certain parameters in
        the form of "weights" (ie weight matrices A and b in equation Ax + b).

        Call this method to get a tuple of the A and b mathematical matrices.

        Args:
            layer (NDArray): the layer of a pretrained neural network model

        Returns:
            Tuple[NDArray, NDArray]: the weights in matrix A and b
        """
        raise NotImplementedError
