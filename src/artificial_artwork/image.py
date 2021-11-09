from abc import ABC, abstractproperty
from typing import List, Callable

import attr
import numpy as np
from numpy.typing import NDArray

ImageLoaderFunctionType = Callable[[str], NDArray]


# class CONFIG:
#     IMAGE_WIDTH = 400
#     IMAGE_HEIGHT = 300
#     COLOR_CHANNELS = 3
#     NOISE_RATIO = 0.6
#     MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))



@attr.s
class Image:
    """An image loaded from disk, represented as a 2D mathematical matrix."""
    file_path: str = attr.ib()
    matrix: NDArray = attr.ib(default=None)

    @classmethod
    def load_matrix(cls, image_path: str, image_loader: ImageLoaderFunctionType):
        return image_loader(image_path)


class ImageProcessingConfigInterface(ABC):

    @abstractproperty
    def image_width(self) -> int:
        raise NotImplementedError
    
    @abstractproperty
    def image_height(self) -> int:
        raise NotImplementedError
    
    @abstractproperty
    def color_channels(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def noise_ratio(self) -> float:
        raise NotImplementedError

    @abstractproperty
    def means(self) -> NDArray:
        raise NotImplementedError


@attr.s
class ImageProcessingConfig(ImageProcessingConfigInterface):
    _image_width = attr.ib()
    _image_height = attr.ib()
    _color_channels = attr.ib()
    _noise_ratio = attr.ib()
    _means = attr.ib()

    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @property
    def color_channels(self) -> int:
        return self._color_channels

    @property
    def noise_ratio(self) -> float:
        return self._noise_ratio

    @property
    def means(self) -> NDArray:
        return self._means

    @classmethod
    def from_image_dimensions(cls, width=400, height=300) -> ImageProcessingConfigInterface:
        return ImageProcessingConfig(
            width,
            height,
            3,
            0.6,
            np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
        )


@attr.s
class ImageProcessor:
    config: ImageProcessingConfigInterface = attr.ib()

    def reshape_and_normalize_image(self, image: NDArray) -> NDArray:
        """Reshape and normalize the input image (content or style)"""
        # Reshape image to mach expected input of VGG16
        # print('DEBUG', 'Input Image Matrix SHAPE:', image.shape)
        image = np.reshape(image, ((1,) + image.shape))
        # print('DEBUG', 'RESHAPED SHAPE:', image.shape)
        # Substract the mean to match the expected input of VGG16
        # print('DUBEG', 'MEANS SHAPE', self.config.means.shape)
        try:
            return image - self.config.means
        except ValueError as numpy_broadcast_error: 
            raise ConfigMeansShapeMissmatchError(
                f'Image processor configured for {self.config.color_channels} '
                f'color channels, but input image has {image.shape[-1]}. Please'
                'consider changing the config.means array (see self.config) for'
                ' this ImageProcessor.') from numpy_broadcast_error

    def noisy(self, image: NDArray) -> NDArray:
        """Generates a noisy image by adding random noise to the content_image"""
        noise_image = np.random.uniform(-20, 20, (1, self.config.image_height, self.config.image_width, self.config.color_channels)).astype('float32')

        # Set the input_image to be a weighted average of the content_image and a noise_image
        return noise_image * self.config.noise_ratio + image * (1 - self.config.noise_ratio)

    def process(self, pipeline: List[Callable[[NDArray], NDArray]], image: NDArray) -> NDArray:
        if len(pipeline) > 0:
            processor = pipeline[0]
            pipeline = pipeline[1:]
            return self.process(pipeline, processor(image))
        return image


class ConfigMeansShapeMissmatchError(Exception): pass


@attr.s
class ImageFactory:
    image_loader: ImageLoaderFunctionType = attr.ib()
    image_processor: ImageProcessor = attr.ib(default=attr.Factory(lambda: ImageProcessor(ImageProcessingConfig.from_image_dimensions())))

    def from_disk(self, image_path: str, preprocess=True) -> Image:
        matrix = Image.load_matrix(image_path, self.image_loader)
        if preprocess:
            image = Image(image_path,
                self.image_processor.process([self.image_processor.reshape_and_normalize_image], matrix))
        else:
            image = Image(image_path, matrix)
        return image
