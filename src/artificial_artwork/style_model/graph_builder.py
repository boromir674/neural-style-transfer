from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf


class GraphBuilder:

    def __init__(self):
        self.graph = {}
        self._prev_layer = None

    def _build_layer(self, layer_id: str, layer):
        self.graph[layer_id] = layer
        self._prev_layer = layer
        return self

    def input(self, image_specs):
        self.graph = {}
        return self._build_layer('input', tf.Variable(np.zeros((1, image_specs.height, image_specs.width, image_specs.color_channels)), dtype='float32'))

    def avg_pool(self, layer_id: str):
        return self._build_layer(layer_id,
            tf.nn.avg_pool(self._prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

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
        return tf.compat.v1.nn.conv2d(self._prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
