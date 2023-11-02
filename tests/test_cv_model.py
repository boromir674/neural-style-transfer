import os
import typing as t

import pytest

from artificial_artwork.pretrained_model.model_handler_interface import (
    ModelHandlerInterface,
)
from artificial_artwork.production_networks import NetworkDesign


class TestPreTrainedModel(t.Protocol):
    # property/attribute for doing assertions
    expected_layers: t.Tuple[str]  # ie (conv1_1, conv1_2, .., avgpool5)

    id: str  # ie 'vgg', or 'toy' (for unit-testing)
    handler: ModelHandlerInterface


class NSTTestingModelProtocol(t.Protocol):
    pretrained_model: TestPreTrainedModel
    network_design: NetworkDesign


my_dir = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def graph_factory():
    from artificial_artwork.style_model import graph_factory

    return graph_factory


def test_pretrained_model(model: NSTTestingModelProtocol, graph_factory):
    import sys

    from numpy.typing import NDArray

    # TYPE CHECKING
    TensorflowTensor = t.TypeVar(
        "TensorflowTensor",
        # bound=tf.Tensor
    )
    Layer = t.Union[t.Any, TensorflowTensor]

    # env var used to
    env_var: str = model.pretrained_model.handler.environment_variable
    prod_env_var = "AA_VGG_19"
    if env_var == prod_env_var and not os.environ.get(prod_env_var):
        # this behaviour arises suspicions of whether it was intended or not
        # this might someday cause some nasty bug, warn to refactor
        # fire up a warning using pytest
        pytest.warns(
            UserWarning,
            match=f"Env var name as reported by mode handler is {env_var}, but it is not found when quering for ENV VARS. Is this intended?",
        )

    # Read ENV VAR var and load a .mat file with the pretrained model (ie weights)
    layers: NDArray = model.pretrained_model.handler.load_model_layers()

    image_specs = type("ImageSpecs", (), {"width": 400, "height": 300, "color_channels": 3})()

    assert len(layers) == len(model.pretrained_model.expected_layers)
    for i, name in enumerate(model.pretrained_model.expected_layers):
        assert layers[i][0][0][0][0] == name

    # create a new reporter given the loaded layer weights
    model.pretrained_model.handler.reporter = layers
    model_design = type(
        "ModelDesign",
        (),
        {
            "pretrained_model": model.pretrained_model.handler,
            "network_design": model.network_design,
        },
    )
    # given the weight values loaded from the pretrained model, recreate the
    # original architecture of the model (ie vgg19) by instantiating Layers
    # as Tensor Operations (ie ReLU, Avg pooling)
    # For Neurons/Operations such as ReLU, the pretrained model weight values
    # are used and set to Constants (ie tf.constant) throughout the runtime
    graph: t.Dict[str, Layer] = graph_factory.create(image_specs, model_design)

    assert set(graph.keys()) == set(["input"] + list(model.network_design.network_layers))


# test_verify_that_the_default_layers_selected_to_build_layers_on_are_compatible
# with_ones_present_in_the pretrained model (ie vgg layers)


def test_code_builds_layers_expected_to_be_found_in_prod_image_model(
    vgg_layers: t.Tuple[str],  # ie (conv1_1, conv1_2, .., avgpool5)
):
    # GIVEN the catalog of layers selected by default by the algorithm to build
    # (ie wrap with tf tensors)
    from artificial_artwork.production_networks.image_model import (
        LAYERS as DEFAULT_VGG_LAYERS_TO_BUILD,
    )

    # GIVEN the known / expected Layers to be found stored in the Production
    # pretrained Image Model, used by our NST algorithm
    prod_vgg_layers: t.Tuple[str] = vgg_layers
    # layers consist of a unique ID (str) which also indicates the type of
    # Neuron used in the original architecture and possibly Weights resulted
    # from past training, in the form of A, b matrices (ie neuron func:
    # f(X) = Ax + b)

    # WHEN we check all the requested layers, which expect to load weight values
    # (ie for A, b matrices), against the original Image Model layers

    # layers expecting to load weight matrices from pretrained model are for
    # example the Convolutional Layers (ie conv1_1, conv1_2, .., conv5_4)
    # but not the Average Pooling Layers (ie avgpool1, avgpool2, .., avgpool5)
    layer_types_with_weights = {"conv"}
    from artificial_artwork.style_model.graph_factory import LayerMaker

    regex = LayerMaker(None, None).regex

    runtime_layers_with_weights = [
        l
        for l in DEFAULT_VGG_LAYERS_TO_BUILD
        if regex.match(l) and regex.match(l).group(1) in layer_types_with_weights
    ]
    all_conv_layers_requested_recognized = set(runtime_layers_with_weights).issubset(
        set(prod_vgg_layers)
    )

    # THEN all the requested layers with weights, should be found in the original Image Model
    assert (
        all_conv_layers_requested_recognized
    ), f"These requsted layers where not identified: {set(DEFAULT_VGG_LAYERS_TO_BUILD) - set(prod_vgg_layers)}"
