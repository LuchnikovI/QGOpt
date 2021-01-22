import QGOpt.manifolds as m
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as opt
import tensorflow as tf
import math


class LangevinDynamics(opt.OptimizerV2):
    """
    This is unstable experimental stuff!!!
    Riemannian langevin dynamics algorithm.

    Args:
        manifold: object of the class Manifold, marks a particular manifold.
        eps: floating point number. A step size.
            Defaults to 0.01.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'LangevinDynamics'.

    Notes:
        The sampler works only with real valued tf.Variable of shape
        (..., q, p, 2), where (...) -- enumerates manifolds
        (can be either empty or any shaped),
        q and p size of a matrix, the last index marks
        real and imag parts of a matrix
        (0 -- real part, 1 -- imag part)"""

    def __init__(self,
                 manifold,
                 eps=0.01,
                 name="LangevinDynamics"):

        super(LangevinDynamics, self).__init__(name)
        self._set_hyper("eps", eps)
        self.manifold = manifold

    def _resource_apply_dense(self, grad, var):

        # Complex version of grad and var
        complex_var = m.real_to_complex(var)
        complex_grad = m.real_to_complex(grad)

        # learning rate
        eps = tf.cast(self._get_hyper("eps"), dtype=complex_grad.dtype)
        
        # noise
        noise = tf.random.normal(var.shape, dtype=var.dtype)
        noise = m.real_to_complex(noise)
        noise = noise * math.sqrt(eps)

        # search direction
        r = -self.manifold.egrad_to_rgrad(complex_var,
                                          0.5 * eps * complex_grad + noise)

        # New value of var
        new_var = self.manifold.retraction(complex_var, r)

        # Update of var
        var.assign(m.complex_to_real(new_var))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(LangevinDynamics, self).get_config()
        config.update({
            "eps": self._serialize_hyperparameter("eps"),
            "manifold": self.manifold
        })
        return config
