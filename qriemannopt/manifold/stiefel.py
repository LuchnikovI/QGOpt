import tensorflow as tf
from qriemannopt.manifold import base_manifold

class StiefelManifold(base_manifold.Manifold):
    """Class is used to work with Stiefel manifold. It allows performing all
    necessary operations with elements of manifolds direct product and
    tangent spaces for optimization."""
    # TODO check correctness of transport for canonical metric
    
    def __init__(self, retraction='svd',
                 metric='euclidean',
                 transport='projective'):
        """Returns object of class StiefelManifold.
        Args:
            retruction: string specifies type of retraction. Defaults to
            'svd'. Types of retraction is available now: 'svd', 'cayley'.
            
            metric: string specifies type of metric, Defaults to 'euclidean'.
            Types of metrics is available now: 'euclidean', 'canonical'.
            
            transport: string specifies type of vector transport,
            Defaults to 'projective'. Types of vector transport
            is available now: 'projective'."""
        
        list_of_metrics = ['euclidean', 'canonical']
        list_of_retractions = ['svd', 'cayley']
        list_of_transports = ['projective']
        
        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")
        if retraction not in list_of_retractions:
            raise ValueError("Incorrect retraction")
        if transport not in list_of_transports:
            raise ValueError("Incorrect transport")
            
        super(StiefelManifold, self).__init__(retraction, metric, transport)
        
        
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
        if self._metric=='euclidean':
            s_sq = tf.linalg.trace(tf.linalg.adjoint(vec1) @ vec2)[...,
                                  tf.newaxis,
                                  tf.newaxis]
            
        elif self._metric=='canonical':
            G = tf.eye(u.shape[-2], dtype=u.dtype) - u @ tf.linalg.adjoint(u) / 2
            s_sq = tf.linalg.trace(tf.linalg.adjoint(vec1) @ G @ vec2)[...,
                                  tf.newaxis,
                                  tf.newaxis]
        return tf.sqrt(s_sq)
    
    
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
            
        return 0.5 * u @ (tf.linalg.adjoint(u) @ vec -\
                          tf.linalg.adjoint(vec) @ u) +\
        (tf.eye(u.shape[-2], dtype=u.dtype) -\
         u @ tf.linalg.adjoint(u)) @ vec
        
    
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
            
        if self._metric=='euclidean':
            return 0.5 * u @ (tf.linalg.adjoint(u) @ egrad -\
                              tf.linalg.adjoint(egrad) @ u) +\
            (tf.eye(u.shape[-2], dtype=u.dtype) -\
             u @ tf.linalg.adjoint(u)) @ egrad
             
        elif self._metric=='canonical':
            return egrad - u @ tf.linalg.adjoint(egrad) @ u
             
    
    @tf.function
    def retraction(self, u, vec):
        """Transports point via retraction map.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p), point
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, p), vector of 
            direction
        Returns tf.Tensor of shape (..., q, p) new point"""
        
        if self._retraction=='svd':
            new_u = u + vec
            _, v, w = tf.linalg.svd(new_u)
            return v @ tf.linalg.adjoint(w)
        
        elif self._retraction=='cayley':
            W = vec @ tf.linalg.adjoint(u) -\
            0.5 * u @ (tf.linalg.adjoint(u) @ vec @ tf.linalg.adjoint(u))
            W = W - tf.linalg.adjoint(W)
            I = tf.eye(W.shape[-1], dtype=W.dtype)
            return tf.linalg.inv(I - W / 2) @ (I + W / 2) @ u
        
        
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
        if self._transport=='projective':
            new_u = self.retraction(u, vec2)
            return self.proj(new_u, vec1)
    
    
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
        if self._transport=='projective':
            new_u = self.retraction(u, vec2)
            return new_u, self.proj(new_u, vec1)