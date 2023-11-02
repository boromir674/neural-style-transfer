from typing import Tuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


class GraphBuilder:
    """Build an NST Style Network as a Compuational Graph of Tensor Operations.

    Provides 3 methods for building the graph:
        - input
        - relu_conv_2d
        - avg_pool

    Client code must first call the 'input' method to provide the input image
    specifications (ie width, height, color_channels), which is necessary for
    the first layer of the Graph.

    Instances of this class are stateful, so the order in which the methods are
    called matters. The 'input' method must be called first, followed by
    alternating calls to 'relu_conv_2d' and 'avg_pool' methods.

    The Graph is stored in the 'graph' instance attribute, which is a dict of
    layer_id to the layer's output tensor.
    """

    def __init__(self):
        self.__init()

    def __init(self):
        self.graph = {}
        self._prev_layer = None

    def _build_layer(self, layer_id: str, layer):
        self.graph[layer_id] = layer
        self._prev_layer = layer
        return self

    def input(self, image_specs):
        self.__init()
        return self._build_layer(
            "input",
            tf.Variable(
                np.zeros(
                    (1, image_specs.height, image_specs.width, image_specs.color_channels)
                ),
                dtype="float32",
            ),
        )

    def avg_pool(self, layer_id: str):
        return self._build_layer(
            layer_id,
            tf.nn.avg_pool(
                self._prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
            ),
        )

    def relu_conv_2d(self, layer_id: str, layer_weights: Tuple[NDArray, NDArray]):
        """A Relu wrapped around a convolutional layer.

        Will use the layer_id to find weight (for W and b matrices) values in
        the pretrained model (layer).

        Also uses the layer_id to as dict key to the output graph.
        """
        W, b = layer_weights
        return self._build_layer(layer_id, tf.nn.relu(self._conv_2d(W, b)))

    def _conv_2d(self, W: NDArray, b: NDArray):
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return (
            tf.compat.v1.nn.conv2d(
                self._prev_layer, filter=W, strides=[1, 1, 1, 1], padding="SAME"
            )
            + b
        )
