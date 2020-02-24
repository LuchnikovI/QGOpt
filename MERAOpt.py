import tensorflow as tf
import qriemannopt.manifold as m

@tf.function
def adj(A):
    """
    Since tf engineers do not care about complex numbers,
    it is necessery to introduce correct hermitian conjugation.
    """
    return tf.math.conj(tf.linalg.matrix_transpose(A))

class MERAOpt(tf.optimizers.Optimizer):

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

        #Complex version of grad
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