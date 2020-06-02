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
            metric_grads_and_vars: zip(metric, grads, vars)"""

        for mgv in metric_grads_and_vars:
            metric, grad, var = mgv
            rgrad = self.manifold.egrad_to_rgrad(metric, var, grad)
            var.assign(var - self._hyper["learning_rate"] * rgrad)
