import pytest

@pytest.fixture
def toy_model_data():
    """Create a toy Network """
    import numpy as np

    from functools import reduce
    # This data format emulates the format the production pretrained VGG layer
    # IDs are stored in
    model_layers = (
        'conv1_1',
        'relu1',
        'maxpool1',
    )
    convo_w_weights_shape = (3, 3, 3, 4)

    def load_layers(*args):
        """Load Layers of 3-layered Toy Neural Net, emulating prod VGG format.
        
        It emulates what the production implementation (scipy.io.loadmat) does,
        by returning an object following the same interface as the one returned
        by scipy.io.loadmat, when called on the file storing the production
        pretrained VGG model.
        """
        return {
            'layers': [[
                # 1st Layer: conv1_1
                [[[[model_layers[0]], 'unused', [[
                    # 'A' Matrix weights tensor with shape (3, 3, 3, 4) (total nb of values = 3*3*3*4 = 108)
                    # for this toy Conv Layer we set the tensor values to be 1, 2, 3, ... 3 * 3 * 3 * 4 + 1 = 109
                    np.reshape(np.array([i for i in range(1, reduce(lambda i,j: i*j, convo_w_weights_shape)+1)], dtype=np.float32), convo_w_weights_shape),
                    # 'b' bias vector, which here is an array of shape (1,)
                    # for this toy Conv Layer we set the bias value to be 5
                    np.array([5], dtype=np.float32)
                ]]]]],
                # 2nd Layer: relu1
                [[[[model_layers[1]], 'unused', [['W', 'b']]]]],  # these layer weights are not expected to be used, because the layer is not a Conv layer
                # 3rd Layer: maxpool1
                [[[[model_layers[2]], 'unused', [['W', 'b']]]]],  # these layer weights are not expected to be used, because the layer is not a Conv layer
            ]]
        }

    return load_layers, model_layers


@pytest.fixture
def toy_nst_algorithm(toy_model_data, toy_network_design, monkeypatch):
    def _monkeypatch():
        return_toy_layers, _ = toy_model_data
        from artificial_artwork.production_networks import NetworkDesign
        from artificial_artwork.pretrained_model import ModelHandlerFacility
        # equip Handler Facility Facory with the 'vgg' implementation
        from artificial_artwork.pre_trained_models import vgg
        import scipy.io
        monkeypatch.setattr(scipy.io, 'loadmat', return_toy_layers)  # Patch/replace-with-mock
        return type('ToyNSTModel', (), {
                'pretrained_model': type('ToyModelHandlerWrapper', (), {
                    'handler': ModelHandlerFacility.create('vgg'),  
                }),
                'network_design': NetworkDesign(
                    toy_network_design.network_layers,
                    toy_network_design.style_layers,
                    toy_network_design.output_layer,
                )
            })
    return _monkeypatch

# TEST that the _demo script (artificial_artwork._demo) which interfaces with
# the 'cmd_demo' module (which defines the nst's 'demo' CLI subcommand)

def test_code_of_layer_bridging_demo_cli_cmd_and_backend(
    toy_nst_algorithm,
):

    # GIVEN the module that implements the layer which bridges the CLI demo cmd
    # and the backend
    from artificial_artwork._demo import create_algo_runner

    # GIVEN a function that implements a way to mock/monkeypatch the bridge, so
    # that this test case is a unit-test and does not need to integrate with the
    # production vgg image model
    handler = toy_nst_algorithm().pretrained_model.handler  # monkey patch production pretrained weights
    # handler = toy_nst_algorithm.pretrained_model.handler

    # WHEN we execute the Layer-provided function that initializes the NST algo
    backend_objs = create_algo_runner()

    # THEN we are provided with a way to run/start the iterative algorithm,
    # which by now should be configured (ie define Computational Graph Architecture,
    # Tensor Operations, Cost Functions Computations, etc) and ready to run
    assert backend_objs is not None
    assert backend_objs['run'] is not None

    # WHEN we check the Image Model Layers loaded for the NST Algo
    # for nst_layer_id, layer_arr in handler.reporter._layer_id_2_layer.items():
    #     # THEN we see that the Image Model Layers loaded for the NST Algo are
    #     # the same as the ones we expect
    #     assert nst_layer_id in backend_objs['layers']
    #     assert backend_objs['layers'][nst_layer_id] == layer_arr

    # THEN we verify that the Toy Network was loaded (and not the production)
    import tensorflow as tf
    dg = tf.compat.v1.get_default_graph()
    s: str = dg.as_graph_def(
        from_version=None, add_shapes=False
    )
    print(f"\n -- GRAPH:\n {s}")
    ops_list = dg.get_operations()
    print(f"\n -- OPS:\n {ops_list}")
    # _, toy_image_model_network_layers = toy_model_data
    # assert set(handler.reporter._layer_id_2_layer.keys()) == \
    #     set(toy_image_model_network_layers)
