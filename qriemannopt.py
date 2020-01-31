import tensorflow as tf

@tf.function
def stiefel_proj(u, vec):
    """Returns projection of vector on complex valued
    Stiefel manifold.
    Args:
        u: complex valued tf.Tensor of shape (batch_size, q, p),
        batch of isometric matrices.
        vec: complex valued tf.Tensor of shape (batch_size, q, p),
        batch of matrices.
    Returns:
        tf.Tensor of shape (batch_size, q, p), batch of projected matrices."""

    return 0.5 * u @ (tf.linalg.adjoint(u) @ vec - tf.linalg.adjoint(vec) @ u) +\
    (tf.eye(u.shape[1], dtype=u.dtype) - u @ tf.linalg.adjoint(u)) @ vec


class StiefelSGD(tf.optimizers.Optimizer):

    def __init__(self,
               learning_rate=0.01,
               name="StiefelSGD"):
        """Construct a new Stochastic Gradient Descent optimizer on Stiefel
        manifold.
        Comment:
            The StiefelSGD works only with real valued tf.Variable of shape
            (batch_size, q, p, 2), where batch_size -- number of isometric
            matrices, q and p size of an isometric matrix, last index marks
            real and imag parts of an isometric matrix
            (0 -- real part, 1 -- imag part)
        Args:
            learning_rate: floating point number. The learning rate.
            Defaults to 0.01.
            name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'StiefelSGD'."""
        
        super(StiefelSGD, self).__init__(name)
        self._lr = learning_rate
        self._lr_t = self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")


    def _create_slots(self, var_list):

        #StiefelSGD does not require _create_slots
        pass


    def _resource_apply_dense(self, grad, var):

        #Complex version of grad and var
        complex_var = tf.complex(var[..., 0], var[..., 1])
        complex_grad = tf.complex(grad[..., 0], grad[..., 1])

        #tf version of learning rate
        lr_t = tf.cast(self._lr_t, complex_var.dtype)

        #Riemannian gradient
        grad_proj = stiefel_proj(complex_var, complex_grad)

        #Upadte of vars (step and retraction)
        new_var = complex_var - lr_t * grad_proj
        _, u, v = tf.linalg.svd(new_var)
        new_var = u @ tf.linalg.adjoint(v)

        #Update of var
        var.assign(tf.concat([tf.math.real(new_var)[..., tf.newaxis],
                              tf.math.imag(new_var)[..., tf.newaxis]],
                             axis=-1))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(StiefelSGD, self).get_config()
        config.update({
            "learning_rate": self._lr,
        })
        return config