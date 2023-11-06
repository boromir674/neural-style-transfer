from typing import Iterable, Tuple

from attr import define, field

from .style_layer_selector import NSTLayersSelection


# just to be compatible with current mypy
def style_layers(tuples: Iterable[Tuple[str, float]]):
    return NSTLayersSelection.from_tuples(tuples)


@define
class NetworkDesign:
    """Define what Layers to use for a Neural Style Transfer (NST) Algorithm.

    Neural Style Transfer leverages the Layers of an already trained network (ie
    a VGG Image Model) to extract the style of an image. This class defines what
    Layers to use for this purpose.

    Instances of this class encapsulate the information required to define these
    Layers for the NST Algorithm.

    This information covers 2 aspects of NST:
    - the Style Layers
    - the Output Layer.

    Style Layers are the ones believed to model the Style the pretrained model
    has "learned" and are modeled by this class as a sequence of layer (str) IDS
    paired with a normalized weight.

    Output Layer is the layer used to compute the content loss. It is modeled by
    this class as a single layer (str) ID.

    This class is agnostic of the actual Neural Network used for the NST, so it
    can be used with any Neural Network. Thus, this class's constructor expects
    to receive a sequence of IDs (str) that represent the available/actual
    layers of the pretrained model.

    Args:
        network_layers (Tuple[str]): The available/actual layers of the
        pretrained model.
        style_layers (Tuple[Tuple[str, float]]): Sequence of layer (str) IDS
        paired with a normalized weight, selected as Style Layers.
        output_layer (str): The layer used to compute the content loss.
    """

    # tuple of strings, ie ('conv1_1', 'conv1_2', 'avgpool1', 'conv2_1')
    network_layers: Tuple[str]
    style_layers: NSTLayersSelection = field(converter=style_layers)
    output_layer: str

    @classmethod
    def from_default_vgg(cls):
        from .image_model import LAYERS

        STYLE_LAYERS = (
            ("conv1_1", 0.2),
            ("conv2_1", 0.2),
            ("conv3_1", 0.2),
            ("conv4_1", 0.2),
            ("conv5_1", 0.2),
        )
        return NetworkDesign(
            LAYERS,
            STYLE_LAYERS,
            "conv4_2",
        )
