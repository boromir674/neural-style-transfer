import os
import pytest

from artificial_artwork.pretrained_model.model_loader import get_vgg_19_model_path, load_default_model_parameters

my_dir = os.path.dirname(os.path.realpath(__file__))


PRODUCTION_IMAGE_MODEL = os.environ.get('AA_VGG_19', 'PRETRAINED_MODEL_NOT_FOUND')


@pytest.fixture
def model_parameters():
    from artificial_artwork.pretrained_model.model_loader import load_default_model_parameters
    return load_default_model_parameters()


@pytest.fixture
def vgg_layers():
    """Expected layers structure of the vgg image model."""
    from artificial_artwork.pretrained_model.vgg_architecture import LAYERS
    return LAYERS


@pytest.fixture
def style_network_architecture():
    from artificial_artwork.pretrained_model.image_model import LAYERS
    return LAYERS


@pytest.fixture
def graph_factory():
    from artificial_artwork.pretrained_model import graph_factory
    return graph_factory


@pytest.mark.xfail(not os.path.isfile(PRODUCTION_IMAGE_MODEL),
    reason="No file found to load the pretrained image (cv) model.")
def test_pretrained_model(model_parameters, graph_factory, vgg_layers, style_network_architecture):
    layers = model_parameters['layers']

    image_specs = type('ImageSpecs', (), {
        'width': 400,
        'height': 300,
        'color_channels': 3
    })()

    # verify original/loaded neural network has 43 layers
    assert len(layers[0]) == 43

    for i, name in enumerate(vgg_layers):
        assert layers[0][i][0][0][0][0] == name

    graph = graph_factory.create(image_specs, model_parameters=model_parameters)
    assert set(graph.keys()) == set(['input'] + list(style_network_architecture))


@pytest.fixture
def graph_builder():
    from artificial_artwork.pretrained_model.model_loader import GraphBuilder
    return GraphBuilder()


def test_building_layers(graph_builder):
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()
    import numpy as np
    height = 2
    width = 6
    channels = 2
    expected_input_shape = (1, height, width, channels)

    graph_builder.input(width, height, nb_channels=channels)
    # assert previous layer is the 'input' layer we just added/created
    assert tuple(graph_builder._prev_layer.shape) == expected_input_shape
    assert (graph_builder._prev_layer.numpy() - graph_builder.graph['input'].numpy()).all() == 0
    assert graph_builder.graph['input'].numpy().all() == 0

    # create relu(convolution) layer
    W = np.array(np.random.rand(*expected_input_shape[1:], channels), dtype=np.float32)

    b_weight = 6.0
    b = np.array([b_weight], dtype=np.float32)
    graph_builder.relu_conv_2d('convo1', (W, b))

    # assert the previous layer is the relu(convolution) layer we just added
    assert tuple(graph_builder._prev_layer.shape) == expected_input_shape
    assert (graph_builder.graph['convo1'].numpy() - graph_builder._prev_layer.numpy()).all() == 0
    # We expect that the tensor values are equal to the weight because the algorithm initializes input with tf.zeros
    assert (graph_builder.graph['convo1'].numpy() - b).all() == 0


    # create Average Pooling layer
    layer_id = 'avgpool1'
    graph_builder.avg_pool(layer_id)

    # assert previous layer is the layer we just added/created
    expected_avg_pool_shape = (1, 1, 3, 2)
    expected_avg_output = np.array(
        [[[[b_weight, b_weight, b_weight],
            [b_weight, b_weight, b_weight],
            [b_weight, b_weight, b_weight]
        ]]]
    ,dtype=np.float32)
    assert graph_builder.graph[layer_id].numpy().shape == expected_avg_pool_shape
    assert (graph_builder.graph[layer_id].numpy() - graph_builder._prev_layer.numpy()).all() == 0
    assert (graph_builder.graph[layer_id].numpy() - np.array([b_weight])).all() == 0

    for i in range(2):
        for c in range(2):
            assert graph_builder._prev_layer[0][0][i][c] == graph_builder.graph[layer_id][0][0][i][c]
            assert graph_builder._prev_layer[0][0][i][c] == expected_avg_output[0][0][i][c]
