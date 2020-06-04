import tensorflow as tf


def _to_real_matrix(A):
    """Returns the real representation of an element from
    a complex manifold.
    Args:
        A: complex valued tf tensor of shape (..., q, p)
    Return:
        real valued tf tensor of shape (..., 2 * q, 2 * p)"""

    dim1, dim2 = A.shape[-2:]
    real = tf.math.real(A)
    imag = tf.math.imag(A)
    A_real = tf.concat([tf.concat([real, imag], axis=-1),
                       tf.concat([-imag, real], axis=-1)],
                       axis=-2)
    return A_real


def _to_complex_matrix(A):
    """Returns an element from a complex manifold reconstructed
    from its real representation.
    Args:
        A: real valued tf tensor of shape (..., 2 * q, 2 * p)
    Return:
        complex valued tf tensor of shape (..., q, p)"""

    dim1, dim2 = A.shape[-2:]
    dim1 = int(dim1 / 2)
    dim2 = int(dim2 / 2)
    A_imag = tf.reshape(A, A.shape[:-2] + (2, dim1, 2, dim2))
    A_imag = tf.complex(A_imag[..., 0, :, 0, :], A_imag[..., 0, :, 1, :])
    return A_imag


def _transform_metric(metric):
    """Returns the alternative representation of a metric.
    Args:
        metric: real valued tf tensor of shape (..., q, p, 2, q, p, 2),
        metric
    Return:
        real valued tf tensor of shape (..., 4 * q * p, 4 * q * p),
        the real square matrix representing a metric."""

    dim1, dim2 = metric.shape[-6:-4]
    shape = metric.shape[:-6]
    length = len(shape)

    flip_matrix = tf.constant([[0, 1], [-1, 0]], dtype=metric.dtype)
    metric_flip = tf.tensordot(metric, flip_matrix, axes=1)
    metric_flip = tf.tensordot(metric_flip,
                               flip_matrix, axes=[[length + 2], [0]])
    metric_flip = tf.transpose(metric_flip, tuple(range(length)) +\
                               (length + 0,
                                length + 1,
                                length + 5,
                                length + 2,
                                length + 3,
                                length + 4))
    z = tf.zeros(metric_flip.shape + (1, 1), dtype=metric.dtype)
    ametric = tf.concat([tf.concat([metric[..., tf.newaxis, tf.newaxis], z], axis=-1),
                         tf.concat([z, metric_flip[..., tf.newaxis, tf.newaxis]],
                                   axis=-1)], axis=-2)
    ametric = tf.transpose(ametric, tuple(range(length)) +\
                           (length + 6,
                            length + 0,
                            length + 2,
                            length + 1,
                            length + 7,
                            length + 3,
                            length + 5,
                            length + 4))
    return tf.reshape(ametric, shape + (4 * dim1 * dim2, 4 * dim1 * dim2))


def _adj(A):
    """Correct hermitian adjoint.
    Args:
        A: tf tensor of shape (..., n, m)
    Returns:
        tf tensor of shape (..., m, n), hermitian adjoint matrix"""

    return tf.math.conj(tf.linalg.matrix_transpose(A))
