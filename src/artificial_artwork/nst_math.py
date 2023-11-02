from typing import TypeVar

import tensorflow as tf

# future work: narrow down the type pf matrix argument
# VolumeType = Union[NDArray, Type[tf.python.framework.ops.Tensor]]

VolumeType = TypeVar("VolumeType")


def gram_matrix(matrix: VolumeType) -> VolumeType:
    """Compute the Gram matrix of input 2D matrix.

    In Linear Algebra the Gram matrix G of a set of vectors (u_1, u_2, .. , u_n)
    is the matrix of dot products, whose entries are:

    G_{ij} = u^T_i * u_j = numpy.dot(u_i, u_j)
    OR
    GA = A * A^T

    Uses tenforflow to compute the Gram matrix of the input 2D matrix.

    Args:
        matrix (type): matrix of shape (n_C, n_H * n_W)

    Returns:
        (tf.tensor): Gram matrix of A, of shape (n_C, n_C)
    """
    return tf.matmul(matrix, tf.transpose(matrix))
