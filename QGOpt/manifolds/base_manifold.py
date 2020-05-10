from abc import ABC, abstractmethod


class Manifold(ABC):
    """Base class is used to work with a direct product
    of Riemannian manifolds. An element from a direct product of manifolds
    is described by a complex tf.Tensor with the shape (..., a, b),
    where (...) enumerates manifolds from a direct product (can be either
    empty or not), (a, b) is the shape of a matrix from a particular manifold.
    Args:
        retraction: string specifies type of a retraction.

        metric: string specifies type of a metric.

        transport: string specifies type of a vector transport
    Returns:
        object of the class Manifold
    """

    def __init__(self, retraction,
                 metric):

        self._retraction = retraction
        self._metric = metric

    @abstractmethod
    def inner(self, u, vec1, vec2):
        """Returns scalar product of vectors in tangent space at point u of
        manifolds direct product.
        Args:
            u: point where a tangent space is considered.
            vec1: complex valued tensor, the first vector from tangent space.
            vec2: complex valued tensor, the second vector from tangent space.
        Returns:
            complex valued tensor, manifold wise inner product"""
        pass

    @abstractmethod
    def proj(self, u, vec):
        """Returns projection of a vector on a tangen spaces
        of manifolds direct product.
        Args:
            u: complex valued tf.Tensor, point of a direct product of manifolds
            vec: complex valued tf.Tensor, vector to be projected
        Returns:
            complex valued tf.Tensor, projected vector"""
        pass

    @abstractmethod
    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            u: complex valued tf.Tensor, point from manifolds direct product.
            egrad: complex valued tf.Tensor, Eucledian gradient
        Returns:
            tf.Tensor, Riemannian gradient."""
        pass

    @abstractmethod
    def retraction(self, u, vec):
        """Transports point according a retraction map.
        Args:
            u: complex valued tf.Tensor, a point to be transported.
            vec: complex valued tf.Tensor, a direction vector.
        Returns complex valued tf.Tensor, a new point"""
        pass

    @abstractmethod
    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from point u along vector vec2.
        Args:
            u: complex valued tf.Tensor, an initial point of a manifolds
            direct product
            vec1: complex valued tf.Tensor, a vector to be transported
            vec2: complex valued tf.Tensor, a direction vector.
        Returns:
            complex valued tf.Tensor, a new vector"""
        pass

    @abstractmethod
    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport simultaneously.
        Args:
            u: complex valued tf.Tensor, an initial point of a manifolds direct
            product
            vec1: complex valued tf.Tensor, a vector to be transported
            vec2: complex valued tf.Tensor, a direction vector.
        Returns:
            two complex valued tf.Tensor,
            a new point and a transported vector"""
        pass
