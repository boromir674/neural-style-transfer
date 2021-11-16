import pytest
import tensorflow as tf


@pytest.fixture
def gram_matrix():
    from artificial_artwork.nst_math import gram_matrix
    return gram_matrix


def test_gram_matrix_computation(gram_matrix, session):

    test_data = ([3, 2*1], 1, 4, [
        [15.615461, 12.248833, -29.87157],
        [12.248833, 10.877857, -19.879116],
        [-29.87157, -19.879116, 67.08007],
    ])

    with session(1) as test:
        A = tf.compat.v1.random_normal(
            test_data[0],
            mean=test_data[1],
            stddev=test_data[2]
        )

        GA = gram_matrix(A)
        array = GA.eval()

        assert all([all([abs(value - test_data[3][row_index][column_index]) < 1e-4 for column_index, value in enumerate(row)]) for row_index, row in enumerate(array)])
