from QGOpt.manifolds import base_manifold
import tensorflow as tf
from QGOpt.manifolds.utils import adj


class StiefelManifold(base_manifold.Manifold):
    """The class provides tools for moving points and vectors along the
    complex Stiefel manifold (St(n, p) -- the manifold of complex valued
    isometric matrices of size nxp) and along a direct product of several
    Stiefel manifolds. The geometry of the complex Stiefel manifold is taken
    from
    -----------------------------------------------------------------------
    Sato, H., & Iwai, T. (2013, December). A complex singular value 
    decomposition algorithm based on the Riemannian Newton method.
    In 52nd IEEE Conference on Decision and Control (pp. 2972-2978). IEEE.
    -----------------------------------------------------------------------
    Another paper, which was used as a guide for implementation of this
    class is
    -----------------------------------------------------------------------
    Edelman, A., Arias, T. A., & Smith, S. T. (1998). The geometry of
    algorithms with orthogonality constraints. SIAM journal on Matrix
    Analysis and Applications, 20(2), 303-353.
    -----------------------------------------------------------------------
    Args:
        retraction: string specifies type of retraction. Defaults to
        'svd'. Types of retraction are available: 'svd', 'cayley', 'qr'.

        metric: string specifies type of metric, Defaults to 'euclidean'.
        Types of metrics are available: 'euclidean', 'canonical'."""

    def __init__(self, retraction='svd',
                 metric='euclidean'):

        list_of_metrics = ['euclidean', 'canonical']
        list_of_retractions = ['svd', 'cayley', 'qr']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")
        if retraction not in list_of_retractions:
            raise ValueError("Incorrect retraction")

        super(StiefelManifold, self).__init__(retraction, metric)

    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space of a direct product of manifolds.
        Args:
            u: complex valued tensor of shape (..., q, p),
            an element of manifolds direct product
            vec1: complex valued tensor of shape (..., q, p),
            a vector from a tangent space.
            vec2: complex valued tensor of shape (..., q, p),
            a vector from a tangent spaces.
        Returns:
            complex valued tensor of shape (..., 1, 1),
            manifold wise inner product"""

        if self._metric == 'euclidean':
            s_sq = tf.linalg.trace(adj(vec1) @ vec2)[...,
                                                     tf.newaxis,
                                                     tf.newaxis]
        elif self._metric == 'canonical':
            G = tf.eye(u.shape[-2], dtype=u.dtype) - u @ adj(u) / 2
            s_sq = tf.linalg.trace(adj(vec1) @ G @ vec2)[...,
                                                         tf.newaxis,
                                                         tf.newaxis]
        return tf.cast(tf.math.real(s_sq), dtype=u.dtype)

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

        return 0.5 * u @ (adj(u) @ vec - adj(vec) @ u) +\
                         (tf.eye(u.shape[-2], dtype=u.dtype) -\
                          u @ adj(u)) @ vec

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            an element of a direct product.
            egrad: complex valued tf.Tensor of shape (..., q, p),
            an Euclidean gradient.
        Returns:
            tf.Tensor of shape (..., q, p), the Reimannian gradient."""

        if self._metric == 'euclidean':
            return 0.5 * u @ (adj(u) @ egrad - adj(egrad) @ u) +\
                             (tf.eye(u.shape[-2], dtype=u.dtype) -\
                              u @ adj(u)) @ egrad

        elif self._metric == 'canonical':
            return egrad - u @ adj(egrad) @ u

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
            return v @ adj(w)

        elif self._retraction == 'cayley':
            W = vec @ adj(u) - 0.5 * u @ (adj(u) @ vec @ adj(u))
            W = W - adj(W)
            Id = tf.eye(W.shape[-1], dtype=W.dtype)
            return tf.linalg.inv(Id - W / 2) @ (Id + W / 2) @ u

        elif self._retraction == 'qr':
            new_u = u + vec
            q, r = tf.linalg.qr(new_u)
            diag = tf.linalg.diag_part(r)
            sign = tf.math.sign(diag)[..., tf.newaxis, :]
            return q * sign

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

    def random(self, shape):
        """Returns vector vec from StiefelManifold.
        Usage:
            shape = (4,5,3,2)
            m = manifolds.StiefelManifold()
            vec = m.random(shape)
        Args:
            shape: integer values list (..., q, p),
        Returns:
            complex valued tf.Tensor of shape"""
        vec = tf.complex(tf.random.normal(shape, dtype=tf.float64),
                        tf.random.normal(shape, dtype=tf.float64))
        vec, _ = tf.linalg.qr(vec)
        return vec
