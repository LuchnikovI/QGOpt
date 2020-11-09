import tensorflow as tf


def complex_to_real(tensor):
    """Returns tensor converted from a complex dtype with shape
    (...,) to a real dtype with shape (..., 2), where last index
    marks real [0] and imag [1] parts of a complex valued tensor.

    Args:
        tensor: complex valued tensor of shape (...,).

    Returns:
        real valued tensor of shape (..., 2)."""
    return tf.concat([tf.math.real(tensor)[..., tf.newaxis],
                      tf.math.imag(tensor)[..., tf.newaxis]], axis=-1)


def real_to_complex(tensor):
    """Returns tensor converted from a real dtype with shape
    (..., 2) to complex dtype with shape (...,), where last index
    of a real tensor marks real [0] and imag [1]
    parts of a complex valued tensor.

    Args:
        tensor: real valued tensor of shape (..., 2).

    Returns:
        complex valued tensor of shape (...,)."""
    return tf.complex(tensor[..., 0], tensor[..., 1])
