# TEST that the _demo.py module (artificial_artwork._demo) which interfaces with
# the 'cmd_demo' module (which defines the nst's 'demo' CLI subcommand)


def test_code_of_layer_bridging_demo_cli_cmd_and_backend(
    toy_nst_algorithm,
    test_suite,
    monkeypatch,
):
    from pathlib import Path

    # GIVEN the module that implements the layer which bridges the CLI demo cmd
    # and the backend
    from artificial_artwork import _demo

    # GIVEN a function that implements a way to mock/monkeypatch the bridge, so
    # that this test case is a unit-test and does not need to integrate with the
    # production vgg image model
    handler = toy_nst_algorithm()  # monkey patch production pretrained weights
    # and return a handler designed to handle operations of toy model

    monkeypatch.setattr(_demo, "source_root_dir", Path(test_suite) / "..")

    # WHEN we execute the Layer-provided function that initializes the NST algo
    backend_objs = _demo.create_algo_runner()

    # THEN we are provided with a way to run/start the iterative algorithm,
    # which by now should be configured (ie define Computational Graph Architecture,
    # Tensor Operations, Cost Functions Computations, etc) and ready to run
    assert backend_objs is not None
    assert backend_objs["run"] is not None

    # WHEN we check the Image Model Layers loaded for the NST Algo
    # for nst_layer_id, layer_arr in handler.reporter._layer_id_2_layer.items():
    #     # THEN we see that the Image Model Layers loaded for the NST Algo are
    #     # the same as the ones we expect
    #     assert nst_layer_id in backend_objs['layers']
    #     assert backend_objs['layers'][nst_layer_id] == layer_arr

    # THEN we verify that the Toy Network was loaded (and not the production)
    import tensorflow as tf

    dg = tf.compat.v1.get_default_graph()
    s: str = dg.as_graph_def(from_version=None, add_shapes=False)
    print(f"\n -- GRAPH:\n {s}")
    ops_list = dg.get_operations()
    print(f"\n -- OPS:\n {ops_list}")
    # _, toy_image_model_network_layers = toy_model_data
    # assert set(handler.reporter._layer_id_2_layer.keys()) == \
    #     set(toy_image_model_network_layers)
