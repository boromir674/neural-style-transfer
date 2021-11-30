import os
import pytest


my_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def graph_factory():
    from artificial_artwork.style_model import graph_factory
    return graph_factory


def test_pretrained_model(model, graph_factory):
    layers = model.pretrained_model.handler.load_model_layers()

    image_specs = type('ImageSpecs', (), {
        'width': 400,
        'height': 300,
        'color_channels': 3
    })()

    assert len(layers) == len(model.pretrained_model.expected_layers)
    for i, name in enumerate(model.pretrained_model.expected_layers):
        assert layers[i][0][0][0][0] == name

    model.pretrained_model.handler.reporter = layers
    model_design = type('ModelDesign', (), {
        'pretrained_model': model.pretrained_model.handler,
        'network_design': model.network_design
    })
    graph = graph_factory.create(image_specs, model_design)
    assert set(graph.keys()) == set(['input'] + list(model.network_design.network_layers))
