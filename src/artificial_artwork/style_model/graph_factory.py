import re
from typing import Any, Dict, Iterable, Protocol

import attr
from numpy.typing import NDArray

from .graph_builder import GraphBuilder

ModelParameters = Dict[str, NDArray]


class ImageSpecs(Protocol):
    width: int
    height: int
    color_channels: int


class GraphFactory:
    builder = GraphBuilder()

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

        # Build the Input Layer of the NST Style Network, based on input Image Specs
        cls.builder.input(config)
        LayerMaker(
            cls.builder,
            model_design.pretrained_model.reporter,  # this reporter is able to extract weights
        ).make_layers(
            model_design.network_design.network_layers
        )  # all VGG layers

        return cls.builder.graph


@attr.s
class LayerMaker:
    """Automatically create the NST Style Computational Graph (Neural Net)

    Automatically builds the NST Style Network as a Computational Graph of Tensor
    Operations, given a list of layer_ids (ie names) and a reporter (ie a
    ModelReporter instance) that can provide the weights for each layer_id.

    Automatically creates relu or avg pooling layers, depending on the layer_id.
    """

    graph_builder = attr.ib()
    reporter = attr.ib()

    layer_callbacks = attr.ib(
        init=False,
        default=attr.Factory(
            lambda self: {
                "conv": self._create_relu_layer,
                "avgpool": self._create_avg_pool_layer,
            },
            takes_self=True,
        ),
    )
    regex = attr.ib(init=False, default=re.compile(r"(\w+?)[\d_]*$"))

    def make_layers(self, layers: Iterable[str]):
        for layer_id in layers:
            self._build_layer(layer_id)

    def _build_layer(self, layer_id: str):
        match_instance = self.regex.match(layer_id)
        if match_instance is not None:
            return self.layer_callbacks[match_instance.group(1)](layer_id)
        raise UnknownLayerError(
            f"Failed to construct layer '{layer_id}'. Supported layers are "
            f"[{', '.join((k for k in self.layer_callbacks))}] and regex"
            f"used to parse the layer is '{self.regex.pattern}'"
        )

    def _create_relu_layer(self, layer_id: str):
        """Create Layer Neurons by wrapping a ReLU activation function around a Conv2D Tensor"""
        return self.graph_builder.relu_conv_2d(layer_id, self.reporter.get_weights(layer_id))

    def _create_avg_pool_layer(self, layer_id: str):
        """Create Layer as Neurons performing Average Pooling with padding SAME"""
        return self.graph_builder.avg_pool(layer_id)


class UnknownLayerError(Exception):
    pass
