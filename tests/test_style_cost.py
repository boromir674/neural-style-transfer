import pytest
import tensorflow as tf


@pytest.fixture
def style_cost():
    from neural_style_transfer.cost_computer import NSTLayerStyleCostComputer
    return NSTLayerStyleCostComputer.compute


def test_style_cost_computation(style_cost):
    
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as test:
        tf.compat.v1.set_random_seed(1)
        a_S = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = style_cost(a_S, a_G)
        assert abs(J_style_layer.eval() - 2.2849257) < 1e-5
