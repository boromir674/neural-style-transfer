from typing import Callable, List, Protocol

import attr
from numpy.typing import NDArray

from .image import Image
from .image_processor import ImageProcessor


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

    def from_disk(
        self, image_path: str, pipeline: List[Callable[[NDArray], NDArray]] = [], **kwargs
    ) -> ImageProtocol:
        """Load an image from disk and process it using the pipeline, if given

        Creates an Image instance from an image file on disk. The returned
        instance provides attribute 'file_path' which stores the image's file
        path and attribute 'matrix' which stores the image's pixel values as a
        numpy array. An optional 'pipeline' can be provided, where each element is a function that processes the image's array, e.g. by reshaping it (into a tensor) or normalizing it, etc.

        Args:
            image_path (str): The path to the image file.
            pipeline (List[Callable[[NDArray], NDArray]], optional): A list of functions that process the image's array. Defaults to [].

        Returns:
            ImageProtocol: An instance of the Image class.
        """
        matrix = self.image_loader(image_path, **kwargs)
        matrix = self.image_processor.process(matrix, pipeline)
        return Image(image_path, matrix)

    def from_array(
        self, image_array: NDArray, pipeline: List[Callable[[NDArray], NDArray]] = [], **kwargs
    ) -> ImageProtocol:
        image_array = self.image_processor.process(image_array, pipeline)
        return Image(file_path=None, matrix=image_array)
