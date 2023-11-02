from typing import Protocol

import attr
from numpy.typing import NDArray

from .disk_operations import Disk
from .image import ImageFactory, convert_to_uint8, noisy, reshape_image, subtract

__all__ = ["ImageManager", "noisy", "convert_to_uint8"]


class ImageProtocol(Protocol):
    path: str
    matrix: NDArray


@attr.s
class ImageManager:
    preprocessing_pipeline = attr.ib()

    image_factory: ImageFactory = attr.ib(
        init=False, default=attr.Factory(lambda: ImageFactory(Disk.load_image))
    )
    images_compatible: bool = attr.ib(init=False, default=False)

    _known_types = attr.ib(init=False, default={"content", "style"})

    _content_image: ImageProtocol
    _style_image: ImageProtocol

    @staticmethod
    def default(means):
        return ImageManager(
            [
                lambda matrix: reshape_image(matrix, ((1,) + matrix.shape)),
                lambda matrix: subtract(matrix, means),  # input image must have 3 channels!
            ]
        )

    def __attrs_post_init__(self):
        for image_type in self._known_types:
            setattr(self, f"_{image_type}_image", None)

    def load_from_disk(self, file_path: str, image_type: str):
        if image_type not in self._known_types:
            raise ValueError(
                f"Expected type of image to be one of {self._known_types}; found {image_type}"
            )
        # dynamically call the appropriate content/style setter method
        setattr(
            self,
            f"{image_type}_image",
            self.image_factory.from_disk(file_path, self.preprocessing_pipeline),
        )

    def _set_image(self, image, image_type: str):
        # dynamically set appropriate content/style attribute
        setattr(self, f"_{image_type}_image", image)
        if not (self._content_image is None or self._style_image is None):
            if self._content_image.matrix.shape == self._style_image.matrix.shape:
                self.images_compatible = True
                return
        self.images_compatible = False

    @property
    def content_image(self) -> ImageProtocol:
        return self._content_image

    @content_image.setter
    def content_image(self, image: ImageProtocol) -> None:
        self._set_image(image, "content")

    @property
    def style_image(self) -> ImageProtocol:
        return self._style_image

    @style_image.setter
    def style_image(self, image: ImageProtocol) -> None:
        self._set_image(image, "style")
