import re
from typing import Callable, Dict, Protocol, Any, Iterable
import attr
from numpy.typing import NDArray

# from artificial_artwork.pretrained_model.model_parameters_utils import get_layers, vgg_weights
# from artificial_artwork.pretrained_model.layers_getter import ModelReporter
from .graph_builder import GraphBuilder


ModelParameters = Dict[str, NDArray]

class ImageSpecs(Protocol):
    width: int
    height: int
    color_channels: int


class GraphFactory:
    builder = GraphBuilder()
    # layers_extractor: Callable[[ModelParameters], NDArray] = get_layers

    @classmethod
    def create(cls, config: ImageSpecs, model_design) -> Dict[str, Any]:
        """Create a model for the purpose of 'painting'/generating a picture.

        Creates a Deep Learning Neural Network with most layers having weights
        (aka model parameters) with values extracted from a pre-trained model
        (ie another neural network trained on an image dataset suitably).

        Args:
            config ([type]): [description]
            model_parameters ([type], optional): [description]. Defaults to None.

        Returns:
            Dict[str, Any]: [description]
        """
        # each relu_conv_2d uses pretrained model's layer weights for W and b matrices
        # each average pooling layer uses custom weight values
        # all weights are guaranteed to remain constant (see GraphBuilder._conv_2d method)

        cls.builder.input(config)
        LayerMaker(
            cls.builder,
            model_design.pretrained_model.reporter,
        ).make_layers(model_design.network_design.network_layers)

        return cls.builder.graph

# ModelReporter(cls.layers_extractor(model_design.parameters_loader())

@attr.s
class LayerMaker:
    graph_builder = attr.ib()
    reporter = attr.ib()
    layer_callbacks = attr.ib(init=False, default=attr.Factory(lambda self: {
            'conv': self.relu,
            'avgpool': self.graph_builder.avg_pool
        }, takes_self=True)
    )

    def relu(self, layer_id: str):
        return self.graph_builder.relu_conv_2d(layer_id, self.reporter.get_weights(layer_id))

    def layer(self, layer_id: str):
        matched_string = re.match(r'(\w+?)[\d_]*$', layer_id).group(1)
        return self.layer_callbacks[matched_string](layer_id)

    def make_layers(self, layers: Iterable[str]):
        for layer_id in layers:
            self.layer(layer_id)
