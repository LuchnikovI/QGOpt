from abc import ABC, abstractmethod

class Manifold(ABC):
    """Proper description here"""
    # TODO proper description
    
    def __init__(self, retraction,
                 metric,
                 transport):
        """Returns object of class Manifold.
        Args:
            retruction: string specifies type of retraction.
            
            metric: string specifies type of metric.
            
            transport: string specifies type of vector transport
        Returns:
            object of class Manifold"""
        
        self._retraction = retraction
        self._metric = metric
        self._transport = transport
        
    
    @abstractmethod
    def inner(self, u, vec1, vec2):
        """Returns scalar product of vectors in tangent spaces at points u.
        Args:
            u: points where one consider tangent space
            vec1: complex valued tensor, vector from tangent space.
            vec2: complex valued tensor, vector from tangent space.
        Returns:
            complex valued tensor, scalar products"""
        pass
    
    
    @abstractmethod
    def proj(self, u, vec):
        """Returns projections of vectors on tangen spaces
        of manifolds.
        Args:
            u: complex valued tf.Tensor, points on a manifolds
            vec: complex valued tf.Tensor, vectors to be projected
        Returns:
            complex valued tf.Tensor, projected vectors"""
        pass
        
        
    @abstractmethod
    def egrad_to_rgrad(self, u, egrad):
        """Returns riemannian gradients from euclidean gradients.
        Args:
            u: complex valued tf.Tensor, points on manifolds.
            egrad: complex valued tf.Tensor gradients calculated
            at corresponding manifolds points.
        Returns:
            tf.Tensor, batch of projected reimannian
            gradients."""
        pass
    
    
    @abstractmethod
    def retraction(self, u, vec):
        """Transports manifolds points via retraction map.
        Args:
            u: complex valued tf.Tensor, points to be transported
            vec: complex valued tf.Tensor, vectors of directions
        Returns complex valued tf.Tensor new points on manifolds"""
        pass
    
    
    @abstractmethod
    def vector_transport(self, u, vec1, vec2):
        """Returns vectors vec1 tranported from points u along vec2.
        Args:
            u: complex valued tf.Tensor, initial points on a manifolds
            vec1: complex valued tf.Tensor, vectors to be transported
            vec2: complex valued tf.Tensor, direction vectors.
        Returns:
            complex valued tf.Tensor, transported vectors"""
        pass
    
    
    @abstractmethod
    def retraction_transport(self, u, vec1, vec2):
        """Performs retraction and vector transport at the same time.
        Args:
            u: complex valued tf.Tensor, initial points on a manifolds
            vec1: complex valued tf.Tensor, vectors to be transported
            vec2: complex valued tf.Tensor, direction vectors.
        Returns:
            two complex valued tf.Tensor,
            new points on manifolds and transported vectors"""
        pass