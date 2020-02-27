from tensorflow.python.ops.math_ops import real
from tensorflow.python.ops.math_ops import imag
from tensorflow.python.ops.math_ops import complex as c
from tensorflow.python.ops.array_ops import concat
import tensorflow as tf


def complex_to_real(tensor):
    """Returns tensor converted from complex tensor of shape
    (...,) to real tensor of shape (..., 2), where last index
    marks real [0] and imag [1] parts of complex valued tensor.
    Args:
        tensor: complex valued tf.Tensor of shape (...,)
    Returns:
        real valued tf.Tensor of shape (..., 2)"""
    return concat([real(tensor)[..., tf.newaxis],
                   imag(tensor)[..., tf.newaxis]], axis=-1)


def real_to_complex(tensor):
    """Returns tensor converted from real tensor of shape
    (..., 2) to complex tensor of shape (...,), where last index
    of real tensor marks real [0] and imag [1]
    parts of complex valued tensor.
    Args:
        tensor: real valued tf.Tensor of shape (..., 2)
    Returns:
        complex valued tf.Tensor of shape (...,)"""
    return c(tensor[..., 0], tensor[..., 1])
