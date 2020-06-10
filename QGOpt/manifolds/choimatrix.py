from QGOpt.manifolds import base_manifold
import tensorflow as tf
from QGOpt.manifolds.utils import adj
from QGOpt.manifolds.utils import lyap_symmetric
import math


class ChoiMatrix(base_manifold.Manifold):
    """Class describes manifold of choi matrices. The quotient geometry taken
    from https://arxiv.org/abs/1303.1029.
    Args:
        metric: string specifies type of metric, Defaults to 'euclidean'.
        Types of metrics is available now: 'euclidean'."""

    def __init__(self, metric='euclidean'):

        list_of_metrics = ['euclidean']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")

    def inner(self, u, vec1, vec2):
        """Returns scalar product of vectors in the tangent space at a
        point u of manifolds direct product.
        Args:
            u: point where a tangent space is considered.
            vec1: complex valued tensor of shape (..., n ** 2, n ** 2),
            the first vector from tangent space.
            vec2: complex valued tensor of shape (..., n ** 2, n ** 2),
            the second vector from tangent space.
        Returns:
            complex valued tensor of shape (...),
            manifold wise inner product"""
        prod = tf.reduce_sum(tf.math.conj(vec1) * vec2, axis=(-2, -1))
        prod = tf.math.real(prod)
        prod = tf.cast(prod, dtype=u.dtype)
        return prod

    def proj(self, u, vec):
        """Returns projection of a vector on a tangen spaces
        of manifolds direct product.
        Args:
            u: complex valued tensor of shape (..., n ** 2, n ** 2),
            point of a direct product of manifolds
            vec: complex valued tensor of shape (..., n ** 2, n ** 2),
            vector to be projected
        Returns:
            complex valued tensor of shape (..., n ** 2, n ** 2),
            projected vector"""

        dim = int(math.sqrt(u.shape[-1]))
        shape = u.shape[:-2]

        # projection onto the tangent space of Stiefel manifold
        vec_mod = tf.reshape(vec, shape + (dim, dim ** 3))
        vec_mod = tf.linalg.matrix_transpose(vec_mod)

        u_mod = tf.reshape(u, shape + (dim, dim ** 3))
        u_mod = tf.linalg.matrix_transpose(u_mod)
        vec_mod = vec_mod - 0.5 * u_mod @ (adj(u_mod) @ vec_mod +\
                                           adj(vec_mod) @ u_mod)
        vec_mod = tf.linalg.matrix_transpose(vec_mod)
        vec_mod = tf.reshape(vec_mod, shape + (dim ** 2, dim ** 2))

        # projection onto the horizontal space
        uu = adj(u) @ u
        Omega = lyap_symmetric(uu, adj(u) @ vec_mod - adj(vec_mod) @ u)
        return vec_mod - u @ Omega

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            u: complex valued tensor of shape (..., n ** 2, n ** 2),
            point from manifolds direct product.
            egrad: complex valued tensor of shape (..., n ** 2, n ** 2),
            Eucledian gradient
        Returns:
            complex valued tensor of shape (..., n ** 2, n ** 2)."""

        dim = int(math.sqrt(u.shape[-1]))
        shape = u.shape[:-2]

        # projection onto the tangent space of Stiefel manifold
        vec_mod = tf.reshape(egrad, shape + (dim, dim ** 3))
        vec_mod = tf.linalg.matrix_transpose(vec_mod)

        u_mod = tf.reshape(u, shape + (dim, dim ** 3))
        u_mod = tf.linalg.matrix_transpose(u_mod)
        vec_mod = vec_mod - 0.5 * u_mod @ (adj(u_mod) @ vec_mod +\
                                           adj(vec_mod) @ u_mod)
        vec_mod = tf.linalg.matrix_transpose(vec_mod)
        vec_mod = tf.reshape(vec_mod, shape + (dim ** 2, dim ** 2))
        return vec_mod

    def retraction(self, u, vec):
        """Transports point according a retraction map.
        Args:
            u: complex valued tensor of shape (..., n ** 2, n ** 2),
            a point to be transported.
            vec: complex valued tensor of shape (..., n ** 2, n ** 2),
            a direction vector.
        Returns:
            complex valued tensor of shape (..., n ** 2, n ** 2),
            a new point"""

        dim = int(math.sqrt(u.shape[-1]))
        shape = u.shape[:-2]

        # svd based retraction
        u_new = u + vec
        u_new = tf.reshape(u_new, shape + (dim, dim ** 3))
        _, U, V = tf.linalg.svd(u_new)

        u_new = U @ adj(V)
        u_new = tf.reshape(u_new, shape + (dim ** 2, dim ** 2))
        return u_new

    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from point u along vector vec2.
        Args:
            u: complex valued tensor of shape (..., n ** 2, n ** 2),
            an initial point of a manifolds
            direct product
            vec1: complex valued tensor of shape (..., n ** 2, n ** 2),
            a vector to be transported
            vec2: complex valued tensor of shape (..., n ** 2, n ** 2),
            a direction vector.
        Returns:
            complex valued tensor of shape (..., n ** 2, n ** 2),
            a new vector"""

        u_new = self.retraction(u, vec2)
        return self.proj(u_new, vec1)

    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport simultaneously.
        Args:
            u: complex valued tensor of shape (..., p, p),
            an initial point of a manifolds
            direct product
            vec1: complex valued tensor of shape (..., p, p),
            a vector to be transported
            vec2: complex valued tensor of shape (..., p, p),
            a direction vector.
        Returns:
            two complex valued tensors of shape (..., p, p),
            a new point and a transported vector"""

        u_new = self.retraction(u, vec2)
        return u_new, self.proj(u_new, vec1)
