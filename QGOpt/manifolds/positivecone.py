import tensorflow as tf
from QGOpt.manifolds import base_manifold
from QGOpt.manifolds.utils import adj
from QGOpt.manifolds.utils import _lower
from QGOpt.manifolds.utils import _pull_back_chol
from QGOpt.manifolds.utils import _pull_back_log
from QGOpt.manifolds.utils import _push_forward_log
from QGOpt.manifolds.utils import _push_forward_chol
from QGOpt.manifolds.utils import _f_matrix


class PositiveCone(base_manifold.Manifold):
    """The manifold of Hermitian positive definite matrices of size nxn.
    The manifold is equipped with two types of metric: Log-Cholesky metric
    and Log-Euclidean metric. The geometry of the manifold with Log-Cholesky
    metric is taken from

    Lin, Z. (2019). Riemannian Geometry of Symmetric Positive Definite Matrices
    via Cholesky Decomposition. SIAM Journal on Matrix Analysis and Applications,
    40(4), 1353-1370.

    The geometry of the manifold with Log-Euclidean metric is described for
    instance in

    Huang, Z., Wang, R., Shan, S., Li, X., & Chen, X. (2015, June).
    Log-euclidean metric learning on symmetric positive definite manifold
    with application to image set classification. In International
    conference on machine learning (pp. 720-729).

    Args:
        metric: string specifies type of a metric, Defaults to 'log_cholesky'
            Types of metrics are available: 'log_cholesky', 'log_euclidean.'
        retraction: string specifies type of retraction, Defaults to 'expmap'.
            Types of metrics are available: 'expmap'."""

    def __init__(self, retraction='expmap', metric='log_cholesky'):

        self.rank = 2
        self.quotient = False
        list_of_metrics = ['log_cholesky', 'log_euclidean']
        list_of_retractions = ['expmap']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")
        if retraction not in list_of_retractions:
            raise ValueError("Incorrect retraction")

        super(PositiveCone, self).__init__(retraction, metric)

    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space.

        Args:
            u: complex valued tensor of shape (..., n, n),
                a set of points from the manifold.
            vec1: complex valued tensor of shape (..., n, n),
                a set of tangent vectors from the manifold.
            vec2: complex valued tensor of shape (..., n, n),
                a set of tangent vectors from the manifold.

        Returns:
            complex valued tensor of shape (..., 1, 1),
            manifold wise inner product.

        Note:
            The complexity O(n^3) for both inner products."""

        if self._metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            W = _pull_back_log(vec1, U, lmbd)
            V = _pull_back_log(vec2, U, lmbd)

            prod = tf.math.real(tf.reduce_sum(tf.math.conj(W) * V, axis=(-2, -1), keepdims=True))
            prod = tf.cast(prod, dtype=u.dtype)

            return prod

        elif self._metric == 'log_cholesky':
            u_shape = tf.shape(u)
            L = tf.linalg.cholesky(u)
            inv_L = tf.linalg.inv(L)

            W = _pull_back_chol(vec1, L, inv_L)
            V = _pull_back_chol(vec2, L, inv_L)

            mask = tf.ones(u_shape[-2:], dtype=u.dtype)
            mask = _lower(mask)
            G = mask + tf.linalg.diag(1 / (tf.linalg.diag_part(L) ** 2))
            prod = tf.reduce_sum(tf.math.conj(W) * G * V, axis=(-2, -1))
            prod = tf.math.real(prod)
            prod = prod[..., tf.newaxis, tf.newaxis]
            prod = tf.cast(prod, dtype=u.dtype)

            return prod

    def proj(self, u, vec):
        """Returns projection of vectors on a tangen space
        of the manifold.

        Args:
            u: complex valued tensor of shape (..., n, n),
                a set of points from the manifold.
            vec: complex valued tensor of shape (..., n, n),
                a set of vectors to be projected.

        Returns:
            complex valued tensor of shape (..., n, n),
            a set of projected vectors.

        Note:
            The complexity O(n^2)."""
        return (vec + adj(vec)) / 2

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.

        Args:
            u: complex valued tensor of shape (..., n, n),
                a set of points from the manifold.
            egrad: complex valued tensor of shape (..., n, n),
                a set of Euclidean gradients.

        Returns:
            complex valued tensor of shape (..., n, n),
            the set of Reimannian gradients.

        Note:
            The complexity O(n^3)."""

        if self._metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            f = _f_matrix(lmbd)
            # Riemannian gradient
            E = adj(U) @ ((egrad + adj(egrad)) / 2) @ U
            R = U @ (E * f * f) @ adj(U)
            return R

        elif self._metric == 'log_cholesky':
            L = tf.linalg.cholesky(u)
            
            sym_fl = (egrad + adj(egrad)) @ L
            L_sq_diag = tf.linalg.diag_part(L) ** 2
            term2 = 0.5 * L_sq_diag * tf.linalg.diag_part(sym_fl + adj(sym_fl))
            term2 = tf.linalg.diag(term2)
            term1 = _lower(sym_fl)
            R = term1 + term2

            return _push_forward_chol(R, L)

    def retraction(self, u, vec):
        """Transports a set of points from the manifold via a
        retraction map.

        Args:
            u: complex valued tensor of shape (..., n, n), a set
                of points to be transported.
            vec: complex valued tensor of shape (..., n, n),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, n),
            a set of transported points.

        Note:
            The complexity O(n^3)."""

        if self._metric == 'log_euclidean':
            lmbd, U = tf.linalg.eigh(u)
            # geodesic in S
            Su = U @ tf.linalg.diag(tf.math.log(lmbd)) @ adj(U)
            Svec = _pull_back_log(vec, U, lmbd)
            Sresult = Su + Svec

            return tf.linalg.expm(Sresult)

        elif self._metric == 'log_cholesky':
            L = tf.linalg.cholesky(u)
            inv_L = tf.linalg.inv(L)

            X = _pull_back_chol(vec, L, inv_L)

            inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))

            cholesky_retraction = _lower(L) + _lower(X) +\
                tf.linalg.band_part(L, 0, 0) *\
                tf.exp(tf.linalg.band_part(X, 0, 0) * inv_diag_L)

            return cholesky_retraction @ adj(cholesky_retraction)

    def vector_transport(self, u, vec1, vec2):
        """Returns a vector tranported along an another vector
        via vector transport.

        Args:
            u: complex valued tensor of shape (..., n, n),
                a set of points from the manifold, starting points.
            vec1: complex valued tensor of shape (..., n, n),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, n),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, n),
            a set of transported vectors.

        Note:
            The complexity O(n^3)."""

        if self._metric == 'log_euclidean':
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

        elif self._metric == 'log_cholesky':
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
        """Performs a retraction and a vector transport simultaneously.

        Args:
            u: complex valued tensor of shape (..., n, n),
                a set of points from the manifold, starting points.
            vec1: complex valued tensor of shape (..., n, n),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, n),
                a set of direction vectors.

        Returns:
            two complex valued tensors of shape (..., n, n),
            a set of transported points and vectors."""

        if self._metric == 'log_euclidean':
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

        elif self._metric == 'log_cholesky':
            v = self.retraction(u, vec2)

            L = tf.linalg.cholesky(u)
            inv_L = tf.linalg.inv(L)

            inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))

            X = _pull_back_chol(vec1, L, inv_L)

            K = tf.linalg.cholesky(v)

            L_transport = _lower(X) + tf.linalg.band_part(K, 0, 0) *\
                inv_diag_L * tf.linalg.band_part(X, 0, 0)

            return v, K @ adj(L_transport) + L_transport @ adj(K)

    def random(self, shape, dtype=tf.complex64):
        """Returns a set of points from the manifold generated
        randomly.

        Args:
            shape: tuple of integer numbers (..., n, n),
                shape of a generated matrix.
            dtype: type of an output tensor, can be
                either tf.complex64 or tf.complex128.

        Returns:
            complex valued tensor of shape (..., n, n),
            a generated matrix."""

        list_of_dtypes = [tf.complex64, tf.complex128]

        if dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")
        real_dtype = tf.float64 if dtype == tf.complex128 else tf.float32

        u = tf.complex(tf.random.normal(shape, dtype=real_dtype),
                       tf.random.normal(shape, dtype=real_dtype))
        u = tf.linalg.adjoint(u) @ u
        return u

    def random_tangent(self, u):
        """Returns a set of random tangent vectors to points from
        the manifold.

        Args:
            u: complex valued tensor of shape (..., n, n), points
                from the manifold.

        Returns:
            complex valued tensor, set of tangent vectors to u."""
        
        u_shape = tf.shape(u)
        vec = tf.complex(tf.random.normal(u_shape), tf.random.normal(u_shape))
        vec = tf.cast(vec, dtype=u.dtype)
        vec = self.proj(u, vec)
        return vec

    def is_in_manifold(self, u, tol=1e-5):
        """Checks if a point is in the manifold or not.

        Args:
            u: complex valued tensor of shape (..., n, n),
                a point to be checked.
            tol: small real value showing tolerance.

        Returns:
            bolean tensor of shape (...)."""

        diff_norm = tf.linalg.norm(u - adj(u), axis=(-2, -1))
        u_norm = tf.linalg.norm(u, axis=(-2, -1))
        rel_diff = tf.abs(diff_norm / u_norm)
        herm_mask = tol > rel_diff
        lmbd = tf.math.real(tf.linalg.eigvalsh(u))
        min_lmbd = tf.math.reduce_min(lmbd)
        positivity_mask = min_lmbd > 0
        return tf.math.logical_and(positivity_mask, herm_mask)
