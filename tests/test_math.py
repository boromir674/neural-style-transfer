import pytest
import tensorflow as tf


@pytest.fixture
def gram_matrix():
    from neural_style_transfer.math import gram_matrix
    return gram_matrix


def test_gram_matrix_computation(gram_matrix, session):

    with session(1) as test:
        A = tf.compat.v1.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
        assert str(GA.eval()) == \
            '[[ 15.615461  12.248833 -29.87157 ]\n ' \
            '[ 12.248833  10.877857 -19.879116]\n ' \
            '[-29.87157  -19.879116  67.08007 ]]'
