from QGOpt.manifolds import base_manifold
import tensorflow as tf
from QGOpt.manifolds.utils import adj
from QGOpt.manifolds.utils import lyap_symmetric


class DensityMatrix(base_manifold.Manifold):
    """Class describes manifold of density matrices. The geometry implemented
    here is taken from https://arxiv.org/abs/1303.1029.
    It is also inspired by the Manopt package (www.manopt.org).
    Args:
        metric: string specifies type of metric, Defaults to 'euclidean'.
        Types of metrics is available now: 'euclidean'."""

    def __init__(self, metric='euclidean'):

        list_of_metrics = ['euclidean']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")

    def inner(self, u, vec1, vec2):
        """Returns scalar product of vectors in tangent space at point u of
        manifolds direct product.
        Args:
            u: point where a tangent space is considered.
            vec1: complex valued tensor of shape (..., p, p),
            the first vector from tangent space.
            vec2: complex valued tensor of shape (..., p, p),
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
            u: complex valued tensor of shape (..., p, p),
            point of a direct product of manifolds
            vec: complex valued tensor of shape (..., p, p),
            vector to be projected
        Returns:
            complex valued tensor of shape (..., p, p),
            projected vector"""

        # projection onto the tangent space of ||u||_F = 1
        vec_proj = vec - u * tf.linalg.trace(adj(u) @ vec)[...,
                                            tf.newaxis, tf.newaxis]

        # projection onto the horizontal space
        uu = adj(u) @ u
        Omega = lyap_symmetric(uu, adj(u) @ vec_proj - adj(vec_proj) @ u)
        return vec_proj - u @ Omega

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            u: complex valued tensor of shape (..., p, p),
            point from manifolds direct product.
            egrad: complex valued tensor of shape (..., p, p),
            Eucledian gradient
        Returns:
            complex valued tensor of shape (..., p, p)."""

        rgrad = egrad - u * tf.linalg.trace(adj(u) @ egrad)[...,
                                           tf.newaxis, tf.newaxis]
        return rgrad

    def retraction(self, u, vec):
        """Transports point according a retraction map.
        Args:
            u: complex valued tensor of shape (..., p, p),
            a point to be transported.
            vec: complex valued tensor of shape (..., p, p),
            a direction vector.
        Returns:
            complex valued tensor of shape (..., p, p), a new point"""

        u_new = (u + vec)
        u_new = u_new / tf.linalg.norm(u_new)
        return u_new

    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from point u along vector vec2.
        Args:
            u: complex valued tensor of shape (..., p, p),
            an initial point of a manifolds
            direct product
            vec1: complex valued tensor of shape (..., p, p),
            a vector to be transported
            vec2: complex valued tensor of shape (..., p, p),
            a direction vector.
        Returns:
            complex valued tensor of shape (..., p, p), a new vector"""
        u_new = (u + vec2)
        u_new = u_new / tf.linalg.norm(u_new)
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
        u_new = (u + vec2)
        u_new = u_new / tf.linalg.norm(u_new)
        return u_new, self.proj(u_new, vec1)
