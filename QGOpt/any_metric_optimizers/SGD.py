import QGOpt.manifolds as m
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as opt
import tensorflow as tf


class RSGD():
    """Riemannian gradient descent in any metric. Necessary for implementing of
    optimization schemes like quantum natural gradient.
    Returns a new Riemannian optimizer (NOT inherited from tf optimizer).
    Comment:
        The optimizer works only with real valued tf.Variable of shape
        (..., q, p, 2), where ... -- enumerates manifolds
        (can be either empty or any shaped),
        q and p size of a matrix, the last index marks
        real and imag parts of a matrix
        (0 -- real part, 1 -- imag part)
    Args:
        manifold: object of the class Manifold, marks a particular manifold.
        learning_rate: floating point number. A learning rate.
        Defaults to 0.01."""

    def __init__(self,
                 manifold,
                 learning_rate=0.01):

        self._hyper = {}
        self._hyper["learning_rate"] = learning_rate
        self.manifold = manifold

    def apply_gradients(self, metric_grads_and_vars):
        """Makes optimization step.
        Args:
            metric_grads_and_vars: zip(metric, grads, vars), metric is
            a list of complex valued tensors of shape (..., q, p, q, p),
            grad is a list of real valued tensors of shape (..., q, p, 2),
            vars is a list of real valued tf Variables of shape
            (..., q, p, 2)"""

        lr = tf.complex(self._hyper["learning_rate"], 0)
        for mgv in metric_grads_and_vars:
            metric, grad, var = mgv
            var_c = m.real_to_complex(var)
            grad_c = m.real_to_complex(grad)
            rgrad_c = self.manifold.egrad_to_rgrad(metric, var_c, grad_c)
            rgrad = m.complex_to_real(rgrad_c)
            new_var = self.manifold.retraction(var_c, -lr * rgrad)
            var.assign(new_var)
