import tensorflow as tf
import qriemannopt.manifold as m

class SGD(tf.optimizers.Optimizer):

    def __init__(self,
                 manifold,
                 learning_rate=0.01,
                 momentum=0.0,
                 name="SGD"):
        """Constructs a new Stochastic Gradient Descent optimizer on Stiefel
        manifold.
        Comment:
            The StiefelSGD works only with real valued tf.Variable of shape
            (..., q, p, 2), where ... -- enumerates manifolds 
            (can be either empty or any shaped),
            q and p size of an isometric matrix, the last index marks
            real and imag parts of an isometric matrix
            (0 -- real part, 1 -- imag part)
        Args:
            learning_rate: floating point number. The learning rate.
            Defaults to 0.01.
            name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'StiefelSGD'."""
        
        super(SGD, self).__init__(name)
        self.manifold = manifold
        self._lr = learning_rate
        self._lr_t = self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._momentum = False
        if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self.momentum = momentum


    def _create_slots(self, var_list):
        # create momentum slot if necessary
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")


    def _resource_apply_dense(self, grad, var):

        #Complex version of grad and var
        complex_var = m.real_to_complex(var)
        complex_grad = m.real_to_complex(grad)

        #tf version of learning rate
        lr_t = tf.cast(self._lr_t, complex_var.dtype)

        #Riemannian gradient
        grad_proj = self.manifold.egrad_to_rgrad(complex_var, complex_grad)

        #Upadte of vars (step and retraction)
        if self._momentum:
            #Update momentum
            momentum_var = self.get_slot(var, "momentum")
            momentum_complex = m.real_to_complex(momentum_var)
            momentum_complex = self.momentum * momentum_complex +\
            (1 - self.momentum) * grad_proj

            #NTransport and retruction
            new_var, momentum_complex =\
            m.retraction_transport(complex_var,
                                   lr_t * momentum_complex,
                                   momentum_complex)
            
            momentum_var.assign(m.complex_to_real(momentum_complex))
        else:
            #New value of var
            new_var = self.manifold.retraction(complex_var, lr_t * grad_proj)

        #Update of var
        var.assign(m.complex_to_real(new_var))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(SGD, self).get_config()
        config.update({
            "learning_rate": self._lr,
            "momentum": self.momentum,
            "manifold": self.manifold
        })
        return config


class StiefelAdam(tf.optimizers.Optimizer):

    def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               eps=1e-8,
               ams=False,
               name="StiefelAdam"):
        """Constructs a new Adam optimizer on Stiefel
        manifold.
        Comment:
            The StiefelAdam works only with real valued tf.Variable of shape
            (..., q, p, 2), where ... -- enumerates manifolds 
            (can be either empty or any shaped),
            q and p size of an isometric matrix, the last index marks
            real and imag parts of an isometric matrix
            (0 -- real part, 1 -- imag part)
        Args:
            learning_rate: floating point number. The learning rate.
            Defaults to 0.001.
            beta1: floating point number. exp decay rate for first moment.
            Defaults to 0.9.
            beta2: floating point number. exp decay rate for second moment.
            Defaults to 0.999.
            eps: floating point number. Regularization coeff.
            Defaults to 1e-8.
            ams: boolean number. Use ams or not.
            name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'StiefelAdam'."""
        
        super(StiefelAdam, self).__init__(name)
        self.iter = 0
        self.eps = eps
        self._lr = learning_rate
        self._lr_t = self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        if isinstance(beta1, (int, float)) and (beta1 < 0 or beta1 > 1):
            raise ValueError("`beta1` must be between [0, 1].")
        self.beta1 = beta1
        if isinstance(beta2, (int, float)) and (beta2 < 0 or beta2 > 1):
            raise ValueError("`beta2` must be between [0, 1].")
        self.beta2 = beta2
        self.ams=ams


    def _create_slots(self, var_list):
        # create m and v slots
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.ams:
                self.add_slot(var, "v_hat")


    def _resource_apply_dense(self, grad, var):
        
        self.iter = self.iter + 1

        #Complex version of grad and var
        complex_var = real_to_complex(var)
        complex_grad = real_to_complex(grad)

        #tf version of learning rate
        lr_t = tf.cast(self._lr_t, complex_var.dtype)

        #Riemannian gradient
        grad_proj = egrad_to_rgrad(complex_var, complex_grad)

        #Complex versions of m and v
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        if self.ams:
            v_hat = self.get_slot(var, "v_hat")
            v_hat_complex = real_to_complex(v_hat)
        m_complex = real_to_complex(m)
        v_complex = real_to_complex(v)
        
        #Update m, v and v_hat
        m_complex = self.beta1 * m_complex + (1 - self.beta1) * grad_proj
        v_complex = self.beta2 * v_complex +\
        (1 - self.beta2) * tf.cast(tf.math.abs(grad_proj) ** 2,
        dtype=v_complex.dtype)
        if self.ams:
            v_hat_complex = tf.math.maximum(tf.math.abs(v_complex),
                                    tf.math.abs(v_hat_complex))
            v_hat_complex = tf.cast(v_hat_complex, dtype=v_complex.dtype)
        
        #Bias correction
        lr_corr = lr_t * (sqrt(1 - self.beta2 ** self.iter) /\
                          (1 - self.beta1 ** self.iter))

        #New value of var
        if self.ams:
            new_var = complex_var - lr_corr * m_complex /\
            (tf.math.sqrt(tf.reduce_mean(v_hat_complex,
                                        axis=(-2, -1),
                                        keepdims=True)) + self.eps)
        else:
            new_var = complex_var - lr_corr * m_complex /\
            (tf.math.sqrt(tf.reduce_mean(v_complex,
                                         axis=(-2, -1),
                                         keepdims=True)) + self.eps)
        new_var = retraction(new_var)

        #Vector transport of v and assigning new value of v
        m_complex = vector_transport(new_var, m_complex)
        m.assign(complex_to_real(m_complex))
        
        #Assigning new value of v and v_hat
        v.assign(complex_to_real(v_complex))
        if self.ams:
            v_hat.assign(complex_to_real(v_hat_complex))

        #Update of var
        var.assign(complex_to_real(new_var))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(StiefelAdam, self).get_config()
        config.update({
            "learning_rate": self._lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "iter": self.iter
        })
        return config