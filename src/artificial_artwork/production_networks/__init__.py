from typing import Iterable, Tuple
import attr

from .style_layer_selector import NSTLayersSelection

# just to be compatible with current mypy
def style_layers(tuples: Iterable[Tuple[str, float]]):
    return NSTLayersSelection.from_tuples(tuples)


@attr.s
class NetworkDesign:
    network_layers: Tuple[str] = attr.ib()
    style_layers: NSTLayersSelection = attr.ib(converter=style_layers)
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
