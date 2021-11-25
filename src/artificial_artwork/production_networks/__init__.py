from typing import List, Tuple
import attr

from .style_layer_selector import NSTLayersSelection


@attr.s
class NetworkDesign:
    network_layers: Tuple[str] = attr.ib()
    style_layers: Tuple[float, str] = attr.ib(converter=NSTLayersSelection.from_tuples)
    output_layer: str = attr.ib()

    @classmethod
    def from_default_vgg(cls):
        from .image_model import LAYERS

        STYLE_LAYERS = (
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2),
        )
        return NetworkDesign(
            LAYERS,
            STYLE_LAYERS,
            'conv4_2',
        )
