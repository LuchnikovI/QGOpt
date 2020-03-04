import qriemannopt.manifold as m
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as opt
import tensorflow as tf


def adj(A):
    """Correct adjoint
    Args:
        A: tf.tensor of shape (..., n, m)
    Returns:
        tf tensor of shape (..., m, n), adjoint matrix"""

    return tf.math.conj(tf.linalg.matrix_transpose(A))


class MERAOpt(opt.OptimizerV2):

    def __init__(self,
                 name="Fast"):
        """Constructs a new MERA inspired optimizer.
        Returns:
                object of class MERAOpt"""

        super(MERAOpt, self).__init__(name)

    def _create_slots(self, var_list):
        # MERAOpt does not need slots
        pass

    def _resource_apply_dense(self, grad, var):

        # Complex version of grad
        complex_grad = m.real_to_complex(grad)

        # MERA like update
        _, u, v = tf.linalg.svd(adj(complex_grad))
        var.assign(m.convert.complex_to_real(-v @ adj(u)))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(MERAOpt, self).get_config()
        config.update({
        })
        return config
