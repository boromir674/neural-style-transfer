import os
import pytest

# from artificial_artwork.pretrained_model.model_loader import get_vgg_19_model_path, load_default_model_parameters

my_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def graph_factory():
    from artificial_artwork.style_model import graph_factory
    return graph_factory


# @pytest.mark.xfail(not os.path.isfile(PRODUCTION_IMAGE_MODEL),
#     reason="No file found to load the pretrained image (cv) model.")
def test_pretrained_model(model, graph_factory):
    # model_parameters = pre_trained_model.parameters_loader()
    from artificial_artwork.pretrained_model.model_handler import ModelHandlerFacility
    # model_handler = ModelHandlerFacility.create(model.pretrained_model.id)
    # layers = model_parameters['layers']
    # layers = model_handler.load_model_layers()
    layers = model.pretrained_model.handler.load_model_layers()
    # layers = pre_trained_model.load_model_layers()
    image_specs = type('ImageSpecs', (), {
        'width': 400,
        'height': 300,
        'color_channels': 3
    })()

    # verify original/loaded neural network has 43 layers
    # assert len(layers[0]) == len(pre_trained_model.vgg_layers)

    assert len(layers) == len(model.pretrained_model.expected_layers)
    for i, name in enumerate(model.pretrained_model.expected_layers):
        assert layers[i][0][0][0][0] == name

    from artificial_artwork.style_model.model_design import ModelDesign
    model.pretrained_model.handler.reporter = layers
    model_design = ModelDesign(
        model.pretrained_model.handler,
        model.network_design
    )
    graph = graph_factory.create(image_specs, model_design)
    assert set(graph.keys()) == set(['input'] + list(model.network_design.network_layers))
