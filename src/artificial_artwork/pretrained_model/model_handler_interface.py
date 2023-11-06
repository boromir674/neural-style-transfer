from abc import ABC, abstractmethod
from typing import Dict, Protocol, Tuple

from numpy.typing import NDArray


class PretrainedModelRoutinesCapable(Protocol):
    def load_layers(self, file_path: str) -> NDArray:
        ...

    def get_id(self, layer: NDArray) -> str:
        ...

    def get_layers_dict(self, layers: NDArray) -> Dict[str, NDArray]:
        ...

    def get_weights(self, layer: NDArray) -> Tuple[NDArray, NDArray]:
        ...


class ModelHandlerInterface(ABC):
    @property
    @abstractmethod
    def environment_variable(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_routines(self) -> PretrainedModelRoutinesCapable:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_load_exception_text(self) -> str:
        raise NotImplementedError
