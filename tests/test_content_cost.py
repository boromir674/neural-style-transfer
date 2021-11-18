import pytest
import tensorflow as tf


@pytest.fixture
def compute_cost():
    from artificial_artwork.cost_computer import NSTContentCostComputer
    return NSTContentCostComputer.compute


def test_content_cost_computation(session, compute_cost):
    with session(2) as _test:
        a_C = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_cost(a_C, a_G)
        assert abs(J_content.eval() - 7.0738883) < 1e-5
