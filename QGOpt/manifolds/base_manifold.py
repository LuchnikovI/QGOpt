from abc import ABC, abstractmethod


class Manifold(ABC):
    """Base class is used to walk across manifolds.

    Args:
        retraction: string specifies type of a retraction.
        metric: string specifies type of a metric.

    Returns:
        object of the class Manifold.
    """

    def __init__(self, retraction,
                 metric):

        self._retraction = retraction
        self._metric = metric

    @abstractmethod
    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space.

        Args:
            u: complex valued tensor, a set of points from a manifold.
            vec1: complex valued tensor, a set of tangent vectors
                from a manifold.
            vec2: complex valued tensor, a set of tangent vectors
                from a manifold.

        Returns:
            complex valued tensor, manifold wise inner product"""
        pass

    @abstractmethod
    def proj(self, u, vec):
        """Returns projection of vectors on a tangen space
        of a manifold.

        Args:
            u: complex valued tensor, a set of points from
                a manifold.
            vec: complex valued tensor, a set of vectors
                to be projected.

        Returns:
            complex valued tensor, a set of projected
                vectors."""
        pass

    @abstractmethod
    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.

        Args:
            u: complex valued tensor, a set of points from a manifold.
            egrad: complex valued tensor, a set of Euclidean gradients.

        Returns:
            complex valued tensor, the set of Reimannian gradients."""
        pass

    @abstractmethod
    def retraction(self, u, vec):
        """Transports a set of points from a manifold via a
        retraction map.

        Args:
            u: complex valued tensor, a set of points to be
                transported.
            vec: complex valued tensor, a set of direction
                vectors.

        Returns:
            complex valued tensor, a set of transported
            points."""
        pass

    @abstractmethod
    def vector_transport(self, u, vec1, vec2):
        """Returns a vector tranported along an another vector
        via vector transport.

        Args:
            u: complex valued tensor, a set of points from
                a manifold, starting points.
            vec1: complex valued tensor, a set of vectors
                to be transported.
            vec2: complex valued tensor, a set of direction
                vectors.

        Returns:
            complex valued tensor, a set of transported vectors."""
        pass

    @abstractmethod
    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport simultaneously.

        Args:
            u: complex valued tensor, a set of points from a
                manifold, starting points.
            vec1: complex valued tensor, a set of vectors
                to be transported.
            vec2: complex valued tensor, a set of direction
                vectors.

        Returns:
            two complex valued tensors, a set of transported
            points and vectors."""
        pass

    @abstractmethod
    def random(self, shape):
        """Returns a set of points from the manifold generated
        randomly.

        Args:
            shape: tuple of integer numbers, shape of a
                generated matrix.
            dtype: type of an output tensor, can be either
                tf.complex64 or tf.complex128.

        Returns:
            complex valued tensor, a generated matrix."""
        pass

    @abstractmethod
    def random_tangent(self, u):
        """Returns a set of random tangent vectors to points from
        a manifold.

        Args:
            u: complex valued tensor of shape (..., n, p), points
                from a manifold.

        Returns:
            complex valued tensor, set of tangent vectors to u."""

        pass

    @abstractmethod
    def is_in_manifold(self, u):
        """Checks if a point is in a manifold or not.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a point to be checked.

        Returns:
            bolean tensor of shape (...)."""
        pass
