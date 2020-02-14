import tensorflow as tf
from qriemannopt.manifold import base_manifold

class StiefelManifold(base_manifold.Manifold):
    # TODO proper description
    def __init__(self, retraction='svd',
                 metric='euclidean',
                 transport='projective'):
        """Returns object of class StiefelManifold.
        Args:
            retruction: string specifies type of retraction. Defaults to
            'svd'. Types of retraction is available now: 'svd', 'cayley'.
            
            metric: string specifies type of metric, Defaults to 'euclidean'.
            Types of metrics is available now: 'euclidean'.
            
            transport: string specifies type of vector transport,
            Defaults to 'projective'. Types of vector transport
            is available now: 'projective'."""
        
        list_of_metrics = ['euclidean']
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
        """Returns manifold wise distance between points.
        Args:
            u: complex valued tensor of shape (..., q, p),
            points on Stiefel manifolds
            vec1: complex valued tensor of shape (..., q, p),
            vector from tangent space.
            vec2: complex valued tensor of shape (..., q, p),
            vector from tangent spaces.
        Returns:
            complex valued tensor of shape (...,), distances between points"""
        if self._metric=='euclidean':
            s_sq = tf.linalg.trace(vec1 @ tf.linalg.adjoint(vec2))[...,
                                  tf.newaxis,
                                  tf.newaxis]
        return tf.sqrt(s_sq)
    
    
    @tf.function
    def proj(self, u, vec):
        """Returns projections of vectors on tangen spaces
        of Stiefel manifolds.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            points on a manifolds
            vec: complex valued tf.Tensor of shape (..., q, p),
            vectors to be projected
        Returns:
            complex valued tf.Tensor of shape (..., q, p), projected vectors"""
            
        if self._metric=='euclidean':
            return 0.5 * u @ (tf.linalg.adjoint(u) @ vec -\
                              tf.linalg.adjoint(vec) @ u) +\
            (tf.eye(u.shape[-2], dtype=u.dtype) -\
             u @ tf.linalg.adjoint(u)) @ vec
        
    
    @tf.function
    def egrad_to_rgrad(self, u, egrad):
        """Returns riemannian gradients from euclidean gradients.
        Equivalent to the projection of gradient on tangent
        spaces of Stiefel manifolds
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            points on Stiefel manifolds.
            egrad: complex valued tf.Tensor of shape (..., q, p),
            gradients calculated at corresponding manifolds points.
        Returns:
            tf.Tensor of shape (..., q, p), batch of projected reimannian
            gradients."""
            
        if self._metric=='euclidean':
            return 0.5 * u @ (tf.linalg.adjoint(u) @ egrad -\
                              tf.linalg.adjoint(egrad) @ u) +\
            (tf.eye(u.shape[-2], dtype=u.dtype) -\
             u @ tf.linalg.adjoint(u)) @ egrad
             
    
    @tf.function
    def retraction(self, u, vec):
        """Transports Stiefel manifolds points via retraction map.
        Args:
            v: complex valued tf.Tensor of shape (..., q, p), points
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, p), vectors of 
            directions
        Returns tf.Tensor of shape (..., q, p) new points
        on Stiefel manifolds"""
        
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
        """Returns vectors tranported to a new points u.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            initial points on a manifolds
            vec1: complex valued tf.Tensor of shape (..., q, p),
            vectors to be transported
            vec2: complex valued tf.Tensor of shape (..., q, p),
            direction vectors.
        Returns:
            complex valued tf.Tensor of shape (..., q, p),
            transported vectors"""
        if self._transport=='projective':
            new_u = self.retraction(u, vec2)
            return self.proj(new_u, vec1)
    
    
    @tf.function
    def retraction_transport(self, u, vec1, vec2):
        """Performs retraction and vector transport at the same time.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            initial points on a manifolds
            vec1: complex valued tf.Tensor of shape (..., q, p),
            vectors to be transported
            vec2: complex valued tf.Tensor of shape (..., q, p),
            direction vectors.
        Returns:
            two complex valued tf.Tensor of shape (..., q, p),
            new points on manifolds and transported vectors"""
        if self._transport=='projective':
            new_u = self.retraction(u, vec2)
            return new_u, self.proj(new_u, vec1)
