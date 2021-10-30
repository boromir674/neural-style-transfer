import tensorflow as tf


def gram_matrix(A):
    """Compute the Gram matrix of input 2D matrix A.

    In Linear Algebra the Gram matrix G of a set of vectors (u_1, u_2, .. , u_n)
    is the matrix of dot products, whose entries are:
    
    G_{ij} = u^T_i * u_j = numpy.dot(u_i, u_j)
    OR
    GA = A * A^T

    Uses tenforflow to compute the Gram matrix of the input 2D matrix.

    Args:
        A (type): matrix of shape (n_C, n_H * n_W)
    
    Returns:
        (tf.tensor): Gram matrix of A, of shape (n_C, n_C)
    """
    return tf.matmul(A, tf.transpose(A))    
