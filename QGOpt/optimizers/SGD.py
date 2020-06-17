import QGOpt.manifolds as m
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as opt
import tensorflow as tf


class RSGD(opt.OptimizerV2):
    """Riemannian gradient descent and gradient descent with momentum
    optimizers. Returns a new Riemannian optimizer.

    Args:
        manifold: object of the class Manifold, marks a particular manifold.
        learning_rate: floating point number. A learning rate.
            Defaults to 0.01.
        momentum: floating point value, the momentum. Defaults to 0
            (Standard GD).
        use_nesterov: Boolean value, if True, use Nesterov Momentum. Defaults
            to False.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'RSGD'.

    Notes:
        The optimizer works only with real valued tf.Variable of shape
        (..., q, p, 2), where (...) -- enumerates manifolds
        (can be either empty or any shaped),
        q and p size of a matrix, the last index marks
        real and imag parts of a matrix
        (0 -- real part, 1 -- imag part)"""

    def __init__(self,
                 manifold,
                 learning_rate=0.01,
                 momentum=0.0,
                 use_nesterov=False,
                 name="RSGD"):

        super(RSGD, self).__init__(name)
        self._set_hyper("learning_rate", learning_rate)
        self.manifold = manifold
        self._momentum = False
        self._use_nesterov = use_nesterov

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
        rgrad = self.manifold.egrad_to_rgrad(complex_var, complex_grad)

        # Upadte of vars (step and retruction)
        if self._momentum:

            if self._use_nesterov:

                # Update momentum
                momentum_var = self.get_slot(var, "momentum")
                momentum_complex_old = m.real_to_complex(momentum_var)
                momentum = tf.cast(self._get_hyper("momentum"),
                                   dtype=momentum_complex_old.dtype)
                momentum_complex_new = momentum * momentum_complex_old +\
                    (1 - momentum) * rgrad

                # Transport and retruction
                new_var, momentum_complex =\
                    self.manifold.retraction_transport(complex_var,
                                                       momentum_complex_new,
                                                       -lr * momentum_complex_new -\
                                                       lr * momentum *\
                                                       (momentum_complex_new -\
                                                        momentum_complex_old))

                momentum_var.assign(m.complex_to_real(momentum_complex))

            else:

                # Update momentum
                momentum_var = self.get_slot(var, "momentum")
                momentum_complex = m.real_to_complex(momentum_var)
                momentum = tf.cast(self._get_hyper("momentum"),
                                   dtype=momentum_complex.dtype)
                momentum_complex = momentum * momentum_complex +\
                    (1 - momentum) * rgrad

                # Transport and retruction
                new_var, momentum_complex =\
                    self.manifold.retraction_transport(complex_var,
                                                       momentum_complex,
                                                       -lr * momentum_complex)

                momentum_var.assign(m.complex_to_real(momentum_complex))
        else:

            # New value of var
            new_var = self.manifold.retraction(complex_var, -lr * rgrad)

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
