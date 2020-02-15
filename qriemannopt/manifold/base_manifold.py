from abc import ABC, abstractmethod

class Manifold(ABC):
    """Base class is used to work with a direct product
    of Riemannian manifolds. An element from a direct product of manifolds
    is described by a complex tf.Tensor with a shape (..., a, b),
    where (...) enumerates manifolds from a direct product (can be either
    empty or not), (a, b) is the shape of a matrix from a particular manifold.
    """
    
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
        """Returns scalar product of vectors in tangent space at point u of
        manifolds direct product.
        Args:
            u: point where one considers tangent space.
            vec1: complex valued tensor, vector from tangent space.
            vec2: complex valued tensor, vector from tangent space.
        Returns:
            complex valued tensor, manifold wise inner product"""
        pass
    
    
    @abstractmethod
    def proj(self, u, vec):
        """Returns projection of vector on tangen spaces
        of manifolds direct product.
        Args:
            u: complex valued tf.Tensor, point of a direct product
            vec: complex valued tf.Tensor, vector to be projected
        Returns:
            complex valued tf.Tensor, projected vector"""
        pass
        
        
    @abstractmethod
    def egrad_to_rgrad(self, u, egrad):
        """Returns riemannian gradients from euclidean gradients.
        Args:
            u: complex valued tf.Tensor, points of manifolds direct product.
            egrad: complex valued tf.Tensor gradient calculated
            at corresponding point.
        Returns:
            tf.Tensor, reimannian gradient."""
        pass
    
    
    @abstractmethod
    def retraction(self, u, vec):
        """Transports point via retraction map.
        Args:
            u: complex valued tf.Tensor, point to be transported.
            vec: complex valued tf.Tensor, vector of direction.
        Returns complex valued tf.Tensor new point of manifolds
        direct product"""
        pass
    
    
    @abstractmethod
    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from point u along vector vec2.
        Args:
            u: complex valued tf.Tensor, initial point of a manifolds
            direct product
            vec1: complex valued tf.Tensor, vector to be transported
            vec2: complex valued tf.Tensor, direction vector.
        Returns:
            complex valued tf.Tensor, transported vector"""
        pass
    
    
    @abstractmethod
    def retraction_transport(self, u, vec1, vec2):
        """Performs retraction and vector transport simultaneously.
        Args:
            u: complex valued tf.Tensor, initial point of a manifolds direct
            product
            vec1: complex valued tf.Tensor, vector to be transported
            vec2: complex valued tf.Tensor, direction vector.
        Returns:
            two complex valued tf.Tensor,
            new point and transported vector"""
        pass