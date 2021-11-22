import os
import pytest

from artificial_artwork.pretrained_model.model_loader import get_vgg_19_model_path, load_default_model_parameters

my_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def graph_factory():
    from artificial_artwork.style_model import graph_factory
    return graph_factory


# @pytest.mark.xfail(not os.path.isfile(PRODUCTION_IMAGE_MODEL),
#     reason="No file found to load the pretrained image (cv) model.")
def test_pretrained_model(pre_trained_model, graph_factory):
    model_parameters = pre_trained_model.parameters_loader()
    layers = model_parameters['layers']

    image_specs = type('ImageSpecs', (), {
        'width': 400,
        'height': 300,
        'color_channels': 3
    })()

    # verify original/loaded neural network has 43 layers
    assert len(layers[0]) == len(pre_trained_model.vgg_layers)

    for i, name in enumerate(pre_trained_model.vgg_layers):
        assert layers[0][i][0][0][0][0] == name

    from artificial_artwork.style_model.model_design import NSTModelDesign

    graph = graph_factory.create(image_specs, NSTModelDesign(
        pre_trained_model.network_layers,
        lambda: model_parameters
    ))
    assert set(graph.keys()) == set(['input'] + list(pre_trained_model.network_layers))
