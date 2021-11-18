### Part of this code is due to the MatConvNet team and is used to load the parameters of the pretrained VGG19 model in the notebook ###

import os
import re
from typing import Dict, Tuple, Any, Protocol

import attr
from numpy.typing import NDArray
import numpy as np
import scipy.io
import tensorflow as tf

from .layers_getter import VggLayersGetter
from .image_model import LAYERS as NETWORK_DESIGN


class ImageSpecs(Protocol):
    width: int
    height: int
    color_channels: int


def load_vgg_model_parameters(path: str) -> Dict[str, NDArray]:
    return scipy.io.loadmat(path)


class NoImageModelSpesifiedError(Exception): pass


def get_vgg_19_model_path():
    try:
        return os.environ['AA_VGG_19']
    except KeyError as variable_not_found:
        raise NoImageModelSpesifiedError('No pretrained image model found. '
            'Please download it and set the AA_VGG_19 environment variable with the'
            'path where ou stored the model (*.mat file), to indicate to wher to '
            'locate and load it') from variable_not_found


def load_default_model_parameters():
    path = get_vgg_19_model_path()
    return load_vgg_model_parameters(path)


def get_layers(model_parameters: Dict[str, NDArray]) -> NDArray:
    return model_parameters['layers'][0]


class GraphBuilder:
    
    def __init__(self):
        self.graph = {}
        self._prev_layer = None
    
    def _build_layer(self, layer_id: str, layer):
        self.graph[layer_id] = layer
        self._prev_layer = layer
        return self

    def input(self, width: int, height: int, nb_channels=3, dtype='float32', layer_id='input'):
        self.graph = {}
        return self._build_layer(layer_id, tf.Variable(np.zeros((1, height, width, nb_channels)), dtype=dtype))

    def avg_pool(self, layer_id: str):
        return self._build_layer(layer_id, tf.nn.avg_pool(self._prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

    def relu_conv_2d(self, layer_id: str, layer_weights):
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


@attr.s
class ModelParameters:
    params = attr.ib(default=attr.Factory(load_default_model_parameters))


class GraphFactory:
    builder = GraphBuilder()

    @classmethod
    def weights(cls, layer: NDArray) -> Tuple[NDArray, NDArray]:
        """Get the weights and bias for a given layer of the VGG model."""
            # wb = vgg_layers[0][layer][0][0][2]
        wb = layer[0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        return W, b

    @classmethod
    def create(cls, config: ImageSpecs, model_parameters=None) -> Dict[str, Any]:
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

        vgg_model_parameters = ModelParameters(*list(filter(None, [model_parameters])))

        vgg_layers = get_layers(vgg_model_parameters.params)

        layer_getter = VggLayersGetter(vgg_layers)

        def relu(layer_id: str):
            return cls.builder.relu_conv_2d(layer_id, cls.weights(layer_getter.id_2_layer[layer_id]))

        layer_callbacks = {
            'conv': relu,
            'avgpool': cls.builder.avg_pool
        }

        def layer(layer_id: str):
            matched_string = re.match(r'(\w+?)[\d_]*$', layer_id).group(1)
            return layer_callbacks[matched_string](layer_id)

        ## Build Graph

        # each relu_conv_2d uses pretrained model's layer weights for W and b matrices
        # each average pooling layer uses custom weight values
        # all weights are guaranteed to remain constant (see GraphBuilder._conv_2d method)

        # cls.builder.input(config.image_width, config.image_height, nb_channels=config.color_channels)
        cls.builder.input(config.width, config.height, nb_channels=config.color_channels)
        for layer_id in NETWORK_DESIGN:
            layer(layer_id)

        return cls.builder.graph
