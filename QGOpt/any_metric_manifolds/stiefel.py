import tensorflow as tf
from QGOpt.any_metric_manifolds.utils import _to_real_matrix
from QGOpt.any_metric_manifolds.utils import _to_complex_matrix
from QGOpt.any_metric_manifolds.utils import _transform_metric
from QGOpt.any_metric_manifolds.utils import _adj
from QGOpt.any_metric_manifolds import base_manifold
import QGOpt.manifolds as m


class StiefelManifold(base_manifold.Manifold):
    """Class describes Stiefel manifold with arbitrary metric.
    It allows performing all necessary operations with
    elements of manifolds direct product and
    tangent spaces for optimization. Returns object of class StiefelManifold.
    Args:
        retraction: string specifies type of retraction. Defaults to
        'svd'. Types of retraction is available now: 'svd', 'cayley'."""

    def __init__(self, retraction='svd'):

        list_of_retractions = ['svd', 'cayley']

        if retraction not in list_of_retractions:
            raise ValueError("Incorrect retraction")

        super(StiefelManifold, self).__init__(retraction)

    def inner(self, metric, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space of a direct product of manifolds.
        Args:
            metric: real valued tensor of shape (..., q, p, 2, q, p, 2),
            the metric in a current point
            vec1: complex valued tensor of shape (..., q, p),
            a vector from a tangent space.
            vec2: complex valued tensor of shape (..., q, p),
            a vector from a tangent spaces.
        Returns:
            complex valued tensor of shape (...,),
            manifold wise inner product"""

        dim1, dim2 = vec1.shape[-2:]
        shape = vec1.shape[:-2]
        vec1_real = m.complex_to_real(vec1)
        vec2_real = m.complex_to_real(vec2)
        vec1_resh = tf.reshape(vec1_real, shape + (2 * dim1 * dim2))
        vec2_resh = tf.reshape(vec2_real, shape + (2 * dim1 * dim2))
        metric_resh = tf.reshape(metric, shape + (2 * dim1 * dim2,
                                                  2 * dim1 * dim2))
        return tf.linalg.matvec(vec1_resh,
                                tf.linalg.matvec(metric_resh, vec2_resh))

    def proj(self, u, vec):
        """Returns projection of a vector on a tangen space
        of a direct product of Stiefel manifolds.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            a point of a direct product.
            vec: complex valued tf.Tensor of shape (..., q, p),
            vectors to be projected.
        Returns:
            complex valued tf.Tensor of shape (..., q, p),
            a projected vector"""

        return 0.5 * u @ (_adj(u) @ vec - _adj(vec) @ u) +\
                         (tf.eye(u.shape[-2], dtype=u.dtype) -\
                          u @ _adj(u)) @ vec

    def egrad_to_rgrad(self, metric, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            metric: real valued tf tensor of shape (..., q, p, 2, q, p, 2),
            the metric in a current point.
            u: complex valued tf.Tensor of shape (..., q, p),
            an element of a direct product.
            egrad: complex valued tf.Tensor of shape (..., q, p),
            an Euclidean gradient.
        Returns:
            tf.Tensor of shape (..., q, p), the Reimannian gradient."""

        dim1, dim2 = u.shape[-2:]
        shape = u.shape[:-2]
        length = len(shape)

        real_egrad = _to_real_matrix(egrad)
        real_egrad = tf.reshape(real_egrad, shape + (4 * dim1 * dim2))
        real_u = _to_real_matrix(u)
        rdtype = real_u.dtype

        T = tf.tensordot(tf.eye(2 * dim1, dtype=rdtype),
                         tf.eye(2 * dim2, dtype=rdtype), axes=0)
        T = tf.transpose(T, (2, 0, 1, 3))
        T = tf.reshape(T, (4 * dim1 * dim2, 4 * dim1 * dim2))

        U1 = (real_u @ tf.linalg.matrix_transpose(real_u))[...,
             tf.newaxis,
             tf.newaxis] * tf.eye(2 * dim2, dtype=rdtype)[tf.newaxis,
                       tf.newaxis]
        U2 = real_u[..., tf.newaxis, tf.newaxis] *\
            tf.linalg.matrix_transpose(real_u[...,
                                              tf.newaxis,
                                              tf.newaxis,
                                              :,
                                              :])
        U1 = tf.transpose(U1, tuple(range(length)) + (length, length + 2,
                          length + 1, length + 3))
        U2 = tf.transpose(U2, tuple(range(length)) + (length, length + 2,
                          length + 1, length + 3))
        U1 = tf.reshape(U1, shape + (4 * dim1 * dim2, 4 * dim1 * dim2))
        U2 = tf.reshape(U2, shape + (4 * dim1 * dim2, 4 * dim1 * dim2))

        pi = tf.eye(4 * dim1 * dim2, dtype=rdtype) -\
            0.5 * (U1 + U2 @ T)

        ametric = _transform_metric(metric)
        rgrad = tf.linalg.pinv(pi @ ametric @ pi) @ real_egrad[..., tf.newaxis]
        rgrad = tf.reshape(rgrad, shape + (2 * dim1, 2 * dim2))
        rgrad = _to_complex_matrix(rgrad)

        return rgrad

    def retraction(self, u, vec):
        """Transports a point via a retraction map.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p), a point
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, p), a vector of
            a direction
        Returns tf.Tensor of shape (..., q, p), a new point"""

        if self._retraction == 'svd':
            new_u = u + vec
            _, v, w = tf.linalg.svd(new_u)
            return v @ _adj(w)

        elif self._retraction == 'cayley':
            W = vec @ _adj(u) - 0.5 * u @ (_adj(u) @ vec @ _adj(u))
            W = W - _adj(W)
            Id = tf.eye(W.shape[-1], dtype=W.dtype)
            return tf.linalg.inv(Id - W / 2) @ (Id + W / 2) @ u

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

        new_u = self.retraction(u, vec2)
        return self.proj(new_u, vec1)

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

        new_u = self.retraction(u, vec2)
        return new_u, self.proj(new_u, vec1)
