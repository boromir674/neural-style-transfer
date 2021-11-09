import pytest
import tensorflow as tf


@pytest.fixture
def style_cost():
    from artificial_artwork.cost_computer import NSTLayerStyleCostComputer
    return NSTLayerStyleCostComputer.compute


def test_style_cost_computation(session, style_cost):

    with session(1) as test:

        a_S = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer = style_cost(a_S, a_G)
        assert abs(J_style_layer.eval() - 2.2849257) < 1e-5
