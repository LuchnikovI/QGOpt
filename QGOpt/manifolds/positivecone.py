import tensorflow as tf
from QGOpt.manifolds import base_manifold
from QGOpt.manifolds.utils import adj
from QGOpt.manifolds.utils import _lower
from QGOpt.manifolds.utils import _pull_back_chol
from QGOpt.manifolds.utils import _pull_back_log
from QGOpt.manifolds.utils import _push_forward_log
from QGOpt.manifolds.utils import _f_matrix


class PositiveCone(base_manifold.Manifold):
    """Class describes S++ manifold. It allows performing all
    necessary operations with elements of manifolds direct product and
    tangent spaces for optimization. Returns object of the class PositiveCone
    Args:
        metric: string specifies type of a metric, Defaults to 'log_cholesky'
        Types of metrics are available now: 'log_cholesky', 'log_euclidean'"""

    def __init__(self, metric='log_cholesky'):

        list_of_metrics = ['log_cholesky', 'log_euclidean']
        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")
        self.metric = metric

    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space.
        Args:
            u: complex valued tensor of shape (..., q, q),
            an element of manifolds direct product
            vec1: complex valued tensor of shape (..., q, q),
            a vector from tangent space.
            vec2: complex valued tensor of shape (..., q, q),
            a vector from tangent spaces.
        Returns:
            complex valued tensor of shape (...,),
            the manifold wise inner product"""

        if self.metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            W = _pull_back_log(vec1, U, lmbd)
            V = _pull_back_log(vec2, U, lmbd)

            return tf.linalg.trace(adj(W) @ V)

        elif self.metric == 'log_cholesky':
            L = tf.linalg.cholesky(u)
            inv_L = tf.linalg.inv(L)

            X = _pull_back_chol(vec1, L, inv_L)
            Y = _pull_back_chol(vec2, L, inv_L)

            diag_inner = tf.math.conj(tf.linalg.diag_part(X)) *\
                tf.linalg.diag_part(Y) / (tf.linalg.diag_part(L) ** 2)
            diag_inner = tf.reduce_sum(diag_inner, axis=-1)
            triag_inner = tf.reduce_sum(tf.math.conj(_lower(X)) * _lower(Y),
                                        axis=(-2, -1))

            return diag_inner + triag_inner

    def proj(self, u, vec):
        """Returns projection of vectors on a tangen space
        of direct product of manifolds.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            a point on a manifolds direct product.
            vec: complex valued tf.Tensor of shape (..., q, q),
            a vector to be projected.
        Returns:
            complex valued tf.Tensor of shape (..., q, q), projected vector"""

        return (vec + adj(vec)) / 2

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            an element of direct product.
            egrad: complex valued tf.Tensor of shape (..., q, q),
            an Euclidean gradient.
        Returns:
            tf.Tensor of shape (..., q, q), the Reimannian gradient."""

        if self.metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            f = _f_matrix(lmbd)
            # Riemannian gradient
            E = adj(U) @ ((egrad + adj(egrad)) / 2) @ U
            R = U @ (E * f * f) @ adj(U)
            return R

        elif self.metric == 'log_cholesky':
            n = u.shape[-1]
            dtype = u.dtype
            L = tf.linalg.cholesky(u)

            half = tf.ones((n, n),
                           dtype=dtype) - tf.linalg.diag(tf.ones((n,), dtype))
            G = tf.linalg.band_part(half, -1, 0) +\
                tf.linalg.diag(tf.linalg.diag_part(L) ** 2)

            R = L @ adj(G * (egrad @ L))
            R = 2 * (R + adj(R))

            return R

    def retraction(self, u, vec):
        """Transports a point via retraction map.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q), a point
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, q),
            a direction vector
        Returns tf.Tensor of shape (..., q, q) a new point"""

        if self.metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            # geodesic in S
            Su = U @ tf.linalg.diag(tf.math.log(lmbd)) @ adj(U)
            Svec = _pull_back_log(vec, U, lmbd)
            Sresult = Su + Svec

            return tf.linalg.expm(Sresult)

        elif self.metric == 'log_cholesky':
            L = tf.linalg.cholesky(u)
            inv_L = tf.linalg.inv(L)

            X = _pull_back_chol(vec, L, inv_L)

            inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))

            cholesky_retraction = _lower(L) + _lower(X) +\
                tf.linalg.band_part(L, 0, 0) *\
                tf.exp(tf.linalg.band_part(X, 0, 0) * inv_diag_L)

            return cholesky_retraction @ adj(cholesky_retraction)

    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 transported from a point u along a vector vec2.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            an initial point of a direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            a vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            a direction vector.
        Returns:
            complex valued tf.Tensor of shape (..., q, p),
            a new vector."""

        if self.metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            # geoidesic in S
            Su = U @ tf.linalg.diag(tf.math.log(lmbd)) @ adj(U)
            Svec2 = _pull_back_log(vec2, U, lmbd)
            Sresult = Su + Svec2
            # eig decomposition of a new point from S
            log_new_lmbd, new_U = tf.linalg.eigh(Sresult)
            # new lmbd
            new_lmbd = tf.exp(log_new_lmbd)
            # transported vector
            new_vec1 = _push_forward_log(_pull_back_log(vec1, U, lmbd),
                                         new_U, new_lmbd)

            return new_vec1

        elif self.metric == 'log_cholesky':
            v = self.retraction(u, vec2)

            L = tf.linalg.cholesky(u)
            inv_L = tf.linalg.inv(L)

            inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))

            X = _pull_back_chol(vec1, L, inv_L)

            K = tf.linalg.cholesky(v)

            L_transport = _lower(X) + tf.linalg.band_part(K, 0, 0) *\
                inv_diag_L * tf.linalg.band_part(X, 0, 0)

            return K @ adj(L_transport) + L_transport @ adj(K)

    def retraction_transport(self, u, vec1, vec2):
        """Performs retraction and vector transport at the same time.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            an initial point from a direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            a vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            a direction vector.
        Returns:
            two complex valued tf.Tensor of shape (..., q, p),
            a new point and a new vector."""

        if self.metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            # geoidesic in S
            Su = U @ tf.linalg.diag(tf.math.log(lmbd)) @ adj(U)
            Svec2 = _pull_back_log(vec2, U, lmbd)
            Sresult = Su + Svec2
            # eig decomposition of new point from S
            log_new_lmbd, new_U = tf.linalg.eigh(Sresult)
            # new point from S++
            new_point = new_U @ tf.linalg.diag(tf.exp(log_new_lmbd)) @\
                adj(new_U)
            # new lmbd
            new_lmbd = tf.exp(log_new_lmbd)
            # transported vector
            new_vec1 = _push_forward_log(_pull_back_log(vec1, U, lmbd),
                                         new_U, new_lmbd)

            return new_point, new_vec1

        elif self.metric == 'log_cholesky':
            v = self.retraction(u, vec2)

            L = tf.linalg.cholesky(u)
            inv_L = tf.linalg.inv(L)

            inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))

            X = _pull_back_chol(vec1, L, inv_L)

            K = tf.linalg.cholesky(v)

            L_transport = _lower(X) + tf.linalg.band_part(K, 0, 0) *\
                inv_diag_L * tf.linalg.band_part(X, 0, 0)

            return v, K @ adj(L_transport) + L_transport @ adj(K)

    def random(self, shape):
        """Returns vector vec from PositiveConeManifold.
        Usage:
            shape = (4,5,3,3)
            m = manifolds.PositiveCone()
            vec = m.random(shape)
        Args:
            shape: integer values list (..., q, q),
        Returns:
            complex valued tf.Tensor of shape"""
        vec = tf.complex(tf.random.normal(shape, dtype=tf.float64),
                        tf.random.normal(shape, dtype=tf.float64))
        vec = tf.linalg.adjoint(vec) @ vec
        vec = vec / tf.linalg.trace(vec)
        return vec
