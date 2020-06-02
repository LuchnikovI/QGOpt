from abc import ABC, abstractmethod


class Manifold(ABC):
    """Base class is used to work with a direct product
    of Riemannian manifolds with an arbitrary manifold wise metric.
    An element from a direct product of manifolds
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

    def __init__(self, retraction):

        self._retraction = retraction

    @abstractmethod
    def inner(self, metric, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space of a direct product of manifolds.
        Args:
            metric: complex valued tensor of shape (..., q, p, q, p),
            the metric in a current point
            vec1: complex valued tensor of shape (..., q, p),
            a vector from a tangent space.
            vec2: complex valued tensor of shape (..., q, p),
            a vector from a tangent spaces.
        Returns:
            complex valued tensor of shape (...,),
            manifold wise inner product"""

        pass

    @abstractmethod
    def proj(self, u, vec):
        """Returns projection of a vector on a tangen space
        of a direct product of manifolds.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            a point of a direct product.
            vec: complex valued tf.Tensor of shape (..., q, p),
            vectors to be projected.
        Returns:
            complex valued tf.Tensor of shape (..., q, p),
            a projected vector"""

        pass

    @abstractmethod
    def egrad_to_rgrad(self, metric, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            metric: complex valued tf tensor of shape (..., q, p, q, p),
            the metric in a current point.
            u: complex valued tf.Tensor of shape (..., q, p),
            an element of a direct product.
            egrad: complex valued tf.Tensor of shape (..., q, p),
            an Euclidean gradient.
        Returns:
            tf.Tensor of shape (..., q, p), the Reimannian gradient."""

        pass

    @abstractmethod
    def retraction(self, u, vec):
        """Transports a point via a retraction map.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p), a point
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, p), a vector of
            a direction
        Returns tf.Tensor of shape (..., q, p), a new point"""

    pass

    @abstractmethod
    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from a point u along vec2.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            an initial point of a direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            a vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            a direction vector.
        Returns:
            complex valued tf.Tensor of shape (..., q, p),
            a transported vector."""

        pass

    @abstractmethod
    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport at the same time.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            an initial point from direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            a vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            a direction vector.
        Returns:
            two complex valued tf.Tensor of shape (..., q, p),
            a new point and a new vector."""

        pass
