from QGOpt.manifolds import base_manifold
import tensorflow as tf


def adj(A):
    """Correct hermitian adjoint
    Args:
        A: tf tensor of shape (..., n, m)
    Returns:
        tf tensor of shape (..., m, n), hermitian adjoint matrix"""

    return tf.math.conj(tf.linalg.matrix_transpose(A))


class HermitianManifold(base_manifold.Manifold):
    """Class describes  manifold: X from C[q,q]: X = X*' .
    It allows performing all necessary operations with elements of manifolds
    direct product and tangent spaces for optimization.
    Returns object of class HermitianManifold.
    """

    def __init__(self):
        """ Not implemented yet
        """

    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space of a direct product of manifolds.
        Args:
            u: complex valued tensor of shape (..., q, q),
            an element of manifolds direct product
            vec1: complex valued tensor of shape (..., q, q),
            a vector from a tangent space.
            vec2: complex valued tensor of shape (..., q, q),
            a vector from a tangent spaces.
        Returns:
            complex valued tensor of shape (..., 1, 1),
            manifold wise inner product"""

        return tf.linalg.trace(adj(vec1) @ vec2)[...,
                                                 tf.newaxis,
                                                 tf.newaxis]

    def proj(self, u, vec):
        """Returns projection of a vector to Hermitian space.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            a point of a direct product.
            vec: complex valued tf.Tensor of shape (..., q, q),
            vectors to be projected.
        Returns:
            complex valued tf.Tensor of shape (..., q, q),
            a projected vector"""
        return 0.5*(vec + adj(vec))

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            an element of a direct product.
            egrad: complex valued tf.Tensor of shape (..., q, q),
            an Euclidean gradient.
        Returns:
            tf.Tensor of shape (..., q, q), the Reimannian gradient."""

        return self.proj(u, egrad)

    def retraction(self, u, vec):
        """Transports a point via a retraction map.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q), a point
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, q), a vector of
            a direction
        Returns tf.Tensor of shape (..., q, q), a new point"""
        return u + vec

    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from a point u along vec2.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            an initial point of a direct product.
            vec1: complex valued tf.Tensor of shape (..., q, q),
            a vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, q),
            a direction vector.
        Returns:
            complex valued tf.Tensor of shape (..., q, q),
            a transported vector."""
        new_u = self.retraction(u, vec2)
        return self.proj(new_u, vec1)

    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport at the same time.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            an initial point from direct product.
            vec1: complex valued tf.Tensor of shape (..., q, q),
            a vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, q),
            a direction vector.
        Returns:
            two complex valued tf.Tensor of shape (..., q, q),
            a new point and a new vector."""
        new_u = self.retraction(u, vec2)
        return new_u, self.proj(new_u, vec1)

    def random(self, shape):
        """Returns vector vec from HermitianManifold.
        Usage:
            shape = (4,5,3,3)
            m = manifolds.HermitianManifold()
            vec = m.random(shape)
        Args:
            shape: integer values list (..., q, q),
        Returns:
            complex valued tf.Tensor of shape"""
        vec = tf.complex(tf.random.normal(shape, dtype=tf.float64),
            tf.random.normal(shape, dtype=tf.float64))
        vec = vec + adj(vec)
        vec = vec/tf.norm(vec, ord='fro',axis=[-1,-2])[...,
                                                    tf.newaxis,
                                                    tf.newaxis]
        return vec
