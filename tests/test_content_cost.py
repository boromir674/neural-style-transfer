import pytest
import tensorflow as tf


@pytest.fixture
def compute_cost():
    from neural_style_transfer.cost_computer import NSTContentCostComputer
    return NSTContentCostComputer.compute


# @pytest.fixture
# def activations():


def test_content_cost_computation(compute_cost):
    tf.compat.v1.reset_default_graph()
    # tf.reset_default_graph()

    with tf.compat.v1.Session() as test:
        tf.compat.v1.set_random_seed(2)
        a_C = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_cost(a_C, a_G)
        assert abs(J_content.eval() - 7.0738883) < 1e-5
        # assert round(J_content.eval(), 8) == 7.0738883
