from abc import ABC, abstractmethod
from numpy.typing import NDArray


class DiskInterface(ABC):

    @staticmethod
    @abstractmethod
    def save_image(image: NDArray, file_path: str, format=None) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_image(file_path: str) -> NDArray:
        raise NotImplementedError
