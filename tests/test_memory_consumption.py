import pytest

# TEST Counting the size of total memory consumed by the A and b matrices
# required for Building the Computational Graph is the expected one
# Currently the Production Weight Matrices come from  the pretrained VGG19 model
# and the Toy Weight Matrices come from the Toy Model


# when using pytest as Test Runner
# this test case requires the --run-integration flag to be picked
@pytest.mark.integration
def test_prod_weight_matrices_memory_consumption_is_expected_one():
    # GIVEN the default layers our NST algorithm requires to build as part of
    # its Computational Graph, which require to load weight matrices from the
    # pretrained model (ie A, b matrices)
    from artificial_artwork.production_networks.image_model import (
        LAYERS as DEFAULT_VGG_LAYERS_TO_BUILD,
    )

    layer_types_with_weights = {"conv"}
    from artificial_artwork.style_model.graph_factory import LayerMaker

    regex = LayerMaker(None, None).regex

    runtime_layers_with_weights = [
        l
        for l in DEFAULT_VGG_LAYERS_TO_BUILD
        if regex.match(l) and regex.match(l).group(1) in layer_types_with_weights
    ]

    # GIVEN a method to extract A and b matrices from the loaded layers of a
    # pretrained model
    # Equip the entrypoint with a concrete implementaion tailored
    # to our prod vgg model
    from artificial_artwork.pre_trained_models import vgg
    from artificial_artwork.pretrained_model import ModelHandlerFacility

    # create object to delegate all vgg related operations
    vgg_ops = ModelHandlerFacility.create("vgg")

    # GIVEN all the pretrained model layers are loaded in memory
    _layers = vgg_ops.load_model_layers()
    # the above equips the vgg_ops object with a reporter attribute
    # see src/artificial_artwork/pretrained_model/layers_getter.py

    # GIVEN the total memory requirements in bytes to store the A and b matrices
    expected_mem_consumption = 80097536

    # WHEN we count the total memory consumed by the weight matrices
    runtime_memory_consumption = 0

    # WHEN we extract the weight matrices for the corresponding NST layers that require them (ie conv, but no avgpool)
    for nst_layer_requiring_weights in runtime_layers_with_weights:
        # # extract the A, b matrices from the loaded  pretrained image model
        A_vgg, b_vgg = vgg_ops.reporter.get_weights(nst_layer_requiring_weights)
        runtime_memory_consumption += A_vgg.nbytes + b_vgg.nbytes

    # THEN the total memory consumed by the weight matrices is the expected one
    assert runtime_memory_consumption == expected_mem_consumption
