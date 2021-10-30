import pytest
import tensorflow as tf


@pytest.fixture
def gram_matrix():
    from neural_style_transfer.math import gram_matrix
    return gram_matrix


def test_gram_matrix_computation(gram_matrix):
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as test:
        tf.compat.v1.set_random_seed(1)
        A = tf.compat.v1.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
        assert str(GA.eval()) == \
            '[[ 15.615461  12.248833 -29.87157 ]\n ' \
            '[ 12.248833  10.877857 -19.879116]\n ' \
            '[-29.87157  -19.879116  67.08007 ]]'
