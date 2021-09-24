import os
import attr
from pprint import pprint

from style import StyleModel


my_dir = os.path.dirname(os.path.realpath(__file__))

ART_DIR = 'art_output'
PRETRAINED = 'imagenet-vgg-verydeep-19.mat'


@attr.s
class ArtMaster:
    _instance = None
    _output_dir = attr.ib(init=True, default=os.path.join(my_dir, ART_DIR))
    _content_image = attr.ib(init=True, default='')
    _style_image = attr.ib(init=True, default='')
    _vgg_path = attr.ib(init=True, default=os.path.join(my_dir, PRETRAINED))

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def content_image(self):
        return self._content_image

    @content_image.setter
    def content_image(self, image_path):
        self._content_image = image_path

    @property
    def style_image(self):
        return self._style_image

    @style_image.setter
    def style_image(self, image_path):
        self._style_image = image_path

    def build_style(self, pretrained_model, content_layer, style_layers, verbose=True):
        _ = StyleModel.create_style_model(pretrained_model, content_layer, style_layers)
        if verbose:
            pprint(model, indent=4)
        return _

    def show_content(self):
        pass