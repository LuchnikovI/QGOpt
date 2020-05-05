import QGOpt.manifolds as m
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as opt
import tensorflow as tf


class RSGD(opt.OptimizerV2):
    # TODO proper description

    def __init__(self,
                 manifold,
                 learning_rate=0.01,
                 momentum=0.0,
                 name="RSGD"):
        """Constructs a new Riemannian Stochastic Gradient Descent optimizer
        on a manifold.
        Comment:
            The RSGD works only with real valued tf.Variable of shape
            (..., q, p, 2), where ... -- enumerates manifolds
            (can be either empty or any shaped),
            q and p size of a matrix, the last index marks
            real and imag parts of a matrix
            (0 -- real part, 1 -- imag part)
        Args:
            manifold: object marks particular manifold.
            learning_rate: floating point number. The learning rate.
            Defaults to 0.01.
            name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'RSGD'."""

        super(RSGD, self).__init__(name)
        self._set_hyper("learning_rate", learning_rate)
        self.manifold = manifold
        self._momentum = False

        if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and\
                (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

        self._set_hyper("momentum", momentum)

    def _create_slots(self, var_list):

        # create momentum slot if necessary
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var):

        # Complex version of grad and var
        complex_var = m.real_to_complex(var)
        complex_grad = m.real_to_complex(grad)

        # learning rate
        lr = tf.cast(self._get_hyper("learning_rate"),
                     dtype=complex_grad.dtype)

        # Riemannian gradient
        grad_proj = self.manifold.egrad_to_rgrad(complex_var, complex_grad)

        # Upadte of vars (step and retruction)
        if self._momentum:

            # Update momentum
            momentum_var = self.get_slot(var, "momentum")
            momentum_complex = m.real_to_complex(momentum_var)
            momentum = tf.cast(self._get_hyper("momentum"),
                               dtype=momentum_complex.dtype)
            momentum_complex = momentum * momentum_complex +\
                (1 - momentum) * grad_proj

            # Transport and retruction
            new_var, momentum_complex =\
                self.manifold.retraction_transport(complex_var,
                                                   momentum_complex,
                                                   -lr * momentum_complex)

            momentum_var.assign(m.complex_to_real(momentum_complex))
        else:

            # New value of var
            new_var = self.manifold.retraction(complex_var, -lr * grad_proj)

        # Update of var
        var.assign(m.complex_to_real(new_var))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(RSGD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "manifold": self.manifold
        })
        return config