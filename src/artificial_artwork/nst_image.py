import attr
from .image import ImageFactory
from .disk_operations import Disk



@attr.s
class ImageManager:
    preprocessing_pipeline = attr.ib()
    image_factory: ImageFactory = \
        attr.ib(init=False, default=attr.Factory(lambda: ImageFactory(Disk.load_image)))
    images_compatible: bool = attr.ib(init=False, default=False)
    
    _known_types = attr.ib(init=False, default={'content', 'style'})

    def __attrs_post_init__(self):
        for image_type in self._known_types:
            setattr(self, f'_{image_type}_image', None)

    def load_from_disk(self, file_path: str, image_type: str):
        if image_type not in self._known_types:
            raise ValueError(f'Expected type of image to be one of {self._known_types}; found {image_type}')
        # dynamically call the appropriate content/style setter method
        setattr(self, f'{image_type}_image',
            self.image_factory.from_disk(file_path, self.preprocessing_pipeline
        ))

    def _set_image(self, image, image_type: str):
        # dynamically set appropriate content/style attribute
        setattr(self, f'_{image_type}_image', image)
        if not (self._content_image is None or self._style_image is None):
            if self._content_image.matrix.shape == self._style_image.matrix.shape:
                self.images_compatible = True
                return
        self.images_compatible = False

    @property
    def content_image(self):
        return self._content_image

    @content_image.setter
    def content_image(self, image):
        self._set_image(image, 'content')
    
    @property
    def style_image(self):
        return self._style_image

    @style_image.setter
    def style_image(self, image):
        self._set_image(image, 'style')
