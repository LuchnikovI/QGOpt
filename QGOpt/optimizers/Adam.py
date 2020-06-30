import QGOpt.manifolds as m
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as opt
import tensorflow as tf

from tensorflow.python.keras import initializers
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.keras import backend

import functools
import six


class RAdam(opt.OptimizerV2):
    """Riemannain Adam and AMSGrad optimizers. Returns a new optimizer.

    Args:
        manifold: object of the class Manifold, marks a particular manifold.
        learning_rate: real number. A learning rate. Defaults to 0.05.
        beta1: real number. An exponential decay rate for the first moment.
            Defaults to 0.9.
        beta2: real number. An exponential decay rate for the second moment.
            Defaults to 0.999.
        eps: real number. Regularization coeffitient. Defaults to 1e-8.
        ams: boolean number. Use ams (AMSGrad) or not.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'RAdam'.

    Notes:
        The optimizer works only with real valued tf.Variable of shape
        (..., q, p, 2), where (...) -- enumerates manifolds
        (can be either empty or any shaped),
        q and p the size of a matrix, the last index marks
        real and imag parts of a matrix
        (0 -- real part, 1 -- imag part)"""

    def __init__(self,
                 manifold,
                 learning_rate=0.05,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 ams=False,
                 name="RAdam"):

        super(RAdam, self).__init__(name)
        self.manifold = manifold
        self.iter = 0
        self.eps = eps
        self._set_hyper('learning_rate', learning_rate)

        if isinstance(beta1, (int, float)) and (beta1 < 0 or beta1 > 1):
            raise ValueError("`beta1` must be between [0, 1].")
        self._set_hyper('beta1', beta1)
        if isinstance(beta2, (int, float)) and (beta2 < 0 or beta2 > 1):
            raise ValueError("`beta2` must be between [0, 1].")
        self._set_hyper('beta2', beta2)

        self.ams = ams

    # TODO explain why we update this method
    def add_slot(self, var, slot_name, initializer="zeros",
                 manifold_wise=False):
        rank = self.manifold.rank  # rank of tensot of a manifold
        """Add a new slot variable for `var`."""
        if slot_name not in self._slot_names:
          self._slot_names.append(slot_name)
        var_key = opt._var_key(var)
        slot_dict = self._slots.setdefault(var_key, {})
        weight = slot_dict.get(slot_name, None)
        if weight is None:
          if isinstance(initializer, six.string_types) or callable(initializer):
            initializer = initializers.get(initializer)
            if manifold_wise:
                initial_value = functools.partial(
                    initializer, shape=var.shape[:-rank - 1] + rank * (1,) + (2,),
                    dtype=var.dtype)
            else:
                initial_value = functools.partial(
                    initializer, shape=var.shape, dtype=var.dtype)
          else:
            initial_value = initializer
          strategy = distribute_ctx.get_strategy()
          if not strategy.extended.variable_created_in_scope(var):
            raise ValueError(
                "Trying to create optimizer slot variable under the scope for "
                "tf.distribute.Strategy ({}), which is different from the scope "
                "used for the original variable ({}). Make sure the slot "
                "variables are created under the same strategy scope. This may "
                "happen if you're restoring from a checkpoint outside the scope"
                .format(strategy, var))
    
          with strategy.extended.colocate_vars_with(var):
            weight = tf_variables.Variable(
                name="%s/%s" % (var._shared_name, slot_name),  # pylint: disable=protected-access
                dtype=var.dtype,
                trainable=False,
                initial_value=initial_value)
          backend.track_variable(weight)
          slot_dict[slot_name] = weight
          self._restore_slot_variable(
              slot_name=slot_name, variable=var,
              slot_variable=weight)
          self._weights.append(weight)
        return weight

    def _create_slots(self, var_list):
        # Create m and v slots
        for var in var_list:
            self.add_slot(var, "momentum")
            self.add_slot(var, "v", manifold_wise=True)
            if self.ams:
                self.add_slot(var, "v_hat", manifold_wise=True)

    def _resource_apply_dense(self, grad, var):

        self.iter = self.iter + 1

        # Complex version of grad and var
        complex_var = m.real_to_complex(var)
        complex_grad = m.real_to_complex(grad)

        # learning rate
        lr = tf.cast(self._get_hyper("learning_rate"), complex_grad.dtype)

        # Riemannian gradient
        rgrad = self.manifold.egrad_to_rgrad(complex_var, complex_grad)

        # Complex versions of m and v
        momentum = self.get_slot(var, "momentum")
        v = self.get_slot(var, "v")
        if self.ams:
            v_hat = self.get_slot(var, "v_hat")
            v_hat_complex = m.real_to_complex(v_hat)
        momentum_complex = m.real_to_complex(momentum)
        v_complex = m.real_to_complex(v)

        # Update m, v and v_hat
        beta1 = tf.cast(self._get_hyper("beta1"), dtype=momentum_complex.dtype)
        beta2 = tf.cast(self._get_hyper("beta2"), dtype=momentum_complex.dtype)
        momentum_complex = beta1 * momentum_complex +\
            (1 - beta1) * rgrad
        v_complex = beta2 * v_complex +\
            (1 - beta2) * self.manifold.inner(complex_var,
                                              rgrad,
                                              rgrad)
        if self.ams:
            v_hat_complex = tf.maximum(tf.math.real(v_complex),
                                       tf.math.real(v_hat_complex))
            v_hat_complex = tf.cast(v_hat_complex, dtype=v_complex.dtype)

        # Bias correction
        lr_corr = lr * tf.math.sqrt(1 - beta2 ** self.iter) /\
            (1 - beta1 ** self.iter)

        # New value of var
        if self.ams:
            # Search direction
            search_dir = -lr_corr * momentum_complex /\
                (tf.sqrt(v_hat_complex) + self.eps)
            new_var, momentum_complex =\
                self.manifold.retraction_transport(complex_var,
                                                   momentum_complex,
                                                   search_dir)
        else:
            # Search direction
            search_dir = - lr_corr * momentum_complex /\
                (tf.sqrt(v_complex) + self.eps)
            new_var, momentum_complex =\
                self.manifold.retraction_transport(complex_var,
                                                   momentum_complex,
                                                   search_dir)

        # Assigning new value of momentum
        momentum.assign(m.complex_to_real(momentum_complex))
        # Assigning new value of v and v_hat
        v.assign(m.complex_to_real(v_complex))
        if self.ams:
            v_hat.assign(m.complex_to_real(v_hat_complex))

        # Update of var
        var.assign(m.complex_to_real(new_var))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(RAdam, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta1": self._serialize_hyperparameter("beta1"),
            "beta2": self._serialize_hyperparameter("beta2"),
            "eps": self.eps,
            "iter": self.iter,
            "manifold": self.manifold
        })
        return config
