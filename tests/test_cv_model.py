import os
import pytest

from artificial_artwork.pretrained_model.model_loader import get_vgg_19_model_path, load_default_model_parameters

my_dir = os.path.dirname(os.path.realpath(__file__))


PRODUCTION_IMAGE_MODEL = os.environ.get('AA_VGG_19', 'PRETRAINED_MODEL_NOT_FOUND')


@pytest.fixture
def pre_trained_models(toy_pre_trained_model):
    
    from artificial_artwork.pretrained_model.model_loader import load_default_model_parameters
    from artificial_artwork.pretrained_model.vgg_architecture import LAYERS as vgg_layers
    from artificial_artwork.pretrained_model.image_model import LAYERS as picked_layers

    return {
        # Production vgg pretrained model
        'vgg': type('PretrainedModel', (), {
            'parameters_loader': load_default_model_parameters,
            'vgg_layers': vgg_layers,
            'picked_layers': picked_layers,
        }),
        # Toy simulated pretrained model for (mock) testing
        'toy': type('PretrainedModel', (), {
            'parameters_loader': toy_pre_trained_model['parameters_loader'],
            'vgg_layers': toy_pre_trained_model['model_layers'],
            'picked_layers': toy_pre_trained_model['picked_layers'],
        }),
    }

@pytest.fixture
def pre_trained_model(pre_trained_models):
    import os
    return {
        True: pre_trained_models['vgg'],
        False: pre_trained_models['toy'],
    }[os.path.isfile(PRODUCTION_IMAGE_MODEL)]


@pytest.fixture
def graph_factory():
    from artificial_artwork.pretrained_model import graph_factory
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

    from artificial_artwork.pretrained_model.model_loader import NSTModelDesign

    print('-------------', pre_trained_model.picked_layers)
    graph = graph_factory.create(image_specs, NSTModelDesign(
        pre_trained_model.picked_layers,
        model_parameters
    ))
    assert set(graph.keys()) == set(['input'] + list(pre_trained_model.picked_layers))
