import attr
from typing import Protocol, Any, Callable, List
from numpy.typing import NDArray

from .image_processor import ImageProcessor
from .image import Image


# Define type aliases
class ImageProtocol(Protocol):
    file_path: str
    matrix: NDArray

# define type alias for a callable that takes any number of arguments
ImageLoaderFunctionType = Callable[..., NDArray]


@attr.s
class ImageFactory:
    image_loader: ImageLoaderFunctionType = attr.ib()
    image_processor: ImageProcessor = attr.ib(default=attr.Factory(ImageProcessor))

    def from_disk(self, image_path: str, pipeline: List[Callable[[NDArray], NDArray]]=[], **kwargs) -> ImageProtocol:
        matrix = self.image_loader(image_path, **kwargs)
        matrix = self.image_processor.process(matrix, pipeline)
        return Image(image_path, matrix)
