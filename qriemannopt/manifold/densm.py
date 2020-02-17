import tensorflow as tf
from qriemannopt.manifold import base_manifold

class DensM(base_manifold.Manifold):
    """Class is used to work with Stiefel manifold. It allows performing all
    necessary operations with elements of manifolds direct product and
    tangent spaces for optimization."""
    # TODO check correctness of transport for canonical metric
    
    def __init__(self):
        """Returns object of class StiefelManifold.
        Args:
            retruction: string specifies type of retraction. Defaults to
            'svd'. Types of retraction is available now: 'svd', 'cayley'.
            
            metric: string specifies type of metric, Defaults to 'euclidean'.
            Types of metrics is available now: 'euclidean', 'canonical'.
            
            transport: string specifies type of vector transport,
            Defaults to 'projective'. Types of vector transport
            is available now: 'projective'."""
        
        
    @tf.function
    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        tangent space.
        Args:
            u: complex valued tensor of shape (..., q, p),
            element of manifolds direct product
            vec1: complex valued tensor of shape (..., q, p),
            vector from tangent space.
            vec2: complex valued tensor of shape (..., q, p),
            vector from tangent spaces.
        Returns:
            complex valued tensor of shape (...,),
            manifold wise inner product"""
        
        inv_u = tf.linalg.inv(u)
        return tf.linalg.trace(inv_u @ vec1 @ inv_u @ vec2)
    
    
    @tf.function
    def proj(self, u, vec):
        """Returns projection of vector on tangen space
        of direct product of Stiefel manifolds.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            point of direct product.
            vec: complex valued tf.Tensor of shape (..., q, p),
            vectors to be projected.
        Returns:
            complex valued tf.Tensor of shape (..., q, p), projected vector"""
        
        n = u.shape[-1]
        return 0.5 * (tf.linalg.adjoint(vec) + vec) -\
        (tf.eye(n, dtype=u.dtype) / n) * tf.linalg.trace(vec)
        
    
    @tf.function
    def egrad_to_rgrad(self, u, egrad):
        """Returns riemannian gradient from euclidean gradient.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            element of direct product.
            egrad: complex valued tf.Tensor of shape (..., q, p),
            euclidean gradient.
        Returns:
            tf.Tensor of shape (..., q, p), reimannian gradient."""
            
        n = u.shape[-1]
        return 0.5 * (u @ tf.linalg.adjoint(egrad) @ u + u @ egrad @ u) -\
        (u @ u / n) * tf.linalg.trace(egrad)
             
    
    @tf.function
    def retraction(self, u, vec):
        """Transports point via retraction map.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p), point
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, p), vector of 
            direction
        Returns tf.Tensor of shape (..., q, p) new point"""
        new_u = u + vec
        new_u_evals, new_u_evec = tf.linalg.eigh(new_u)
        new_u_evals = tf.cast(tf.math.abs(new_u_evals), dtype=u.dtype)
        new_u_evals = new_u_evals / tf.reduce_sum(new_u_evals,
                                                  axis=-1,
                                                  keepdims=True)
        new_u = (new_u_evec * new_u_evals[..., tf.newaxis, :]) @\
                tf.linalg.adjoint(new_u_evec)
        
        return new_u
        
        
    @tf.function
    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from point u along vec2.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            initial point of direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            direction vector.
        Returns:
            complex valued tf.Tensor of shape (..., q, p),
            transported vector."""
            
        new_u = self.retraction(u, vec2)
        new_u_evals, new_u_evec = tf.linalg.eigh(new_u)
        u_evals, u_evec = tf.linalg.eigh(u)
        sqrt_u_new = (new_u_evec *\
                      tf.math.sqrt(new_u_evals)[..., tf.newaxis, :]) @\
                      tf.linalg.adjoint(new_u_evec)
        sqrt_u_inv = (u_evec *\
                      1 / tf.math.sqrt(u_evals)[..., tf.newaxis, :]) @\
                      tf.linalg.adjoint(u_evec)
        
        return sqrt_u_new @ sqrt_u_inv @ vec1 @ sqrt_u_inv @ sqrt_u_new
    
    
    @tf.function
    def retraction_transport(self, u, vec1, vec2):
        """Performs retraction and vector transport at the same time.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            initial point from direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            direction vector.
        Returns:
            two complex valued tf.Tensor of shape (..., q, p),
            new point and transported vector."""
        
        new_u = u + vec2
        new_u_evals, new_u_evec = tf.linalg.eigh(new_u)
        new_u_evals = tf.cast(tf.math.abs(new_u_evals), dtype=u.dtype)
        new_u_evals = new_u_evals / tf.reduce_sum(new_u_evals,
                                                  axis=-1,
                                                  keepdims=True)
        new_u = (new_u_evec * new_u_evals[..., tf.newaxis, :]) @\
                tf.linalg.adjoint(new_u_evec)

        u_evals, u_evec = tf.linalg.eigh(u)
        sqrt_u_new = (new_u_evec *\
                      tf.math.sqrt(new_u_evals)[..., tf.newaxis, :]) @\
                      tf.linalg.adjoint(new_u_evec)
        sqrt_u_inv = (u_evec *\
                      1 / tf.math.sqrt(u_evals)[..., tf.newaxis, :]) @\
                      tf.linalg.adjoint(u_evec)
        
        return new_u, sqrt_u_new @ sqrt_u_inv @ vec1 @ sqrt_u_inv @ sqrt_u_new