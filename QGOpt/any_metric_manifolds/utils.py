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


def _real_metric(metric):
    """Returns the real representation of a metric on complex manifold.
    Args:
        metric: complex valued tf tensor of shape (..., q, p, q, p), metric
    Return:
        real valued tf tensor of shape (..., 4 * q * p, 4 * q * p),
        the real square matrix representing a metric."""

    dim1, dim2 = metric.shape[-4:-2]
    shape = metric.shape[:-4]
    length = len(shape)

    real_metric = metric[..., tf.newaxis, tf.newaxis]
    real_metric = tf.concat([tf.concat([tf.math.real(real_metric),
                                        tf.math.imag(real_metric)], axis=-1),
                             tf.concat([-tf.math.imag(real_metric),
                                        tf.math.real(real_metric)], axis=-1)],
                                        axis=-2)
    real_metric = tf.tensordot(real_metric, tf.eye(2, dtype=real_metric.dtype),
                               axes=0)
    real_metric = tf.transpose(real_metric, tuple(range(length)) +\
                               (length + 4,
                                length + 0,
                                length + 6,
                                length + 1,
                                length + 5,
                                length + 2,
                                length + 7,
                                length + 3))
    return tf.reshape(real_metric, shape + (4 * dim1 * dim2, 4 * dim1 * dim2))


def _adj(A):
    """Correct hermitian adjoint.
    Args:
        A: tf tensor of shape (..., n, m)
    Returns:
        tf tensor of shape (..., m, n), hermitian adjoint matrix"""

    return tf.math.conj(tf.linalg.matrix_transpose(A))
