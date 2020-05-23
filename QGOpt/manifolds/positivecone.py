import tensorflow as tf
from QGOpt.manifolds import base_manifold


def adj(A):
    """Returns hermitian adjoint matrix
    Args:
        A: tf.tensor of shape (..., n, m)
    Returns:
        tf tensor of shape (..., m, n), adjoint matrix"""

    return tf.math.conj(tf.linalg.matrix_transpose(A))


def _lower(X):
    """Returns the lower triangular part of a matrix without diagonal part.
    Args:
        X: tf tensor of shape (..., m, m)
    Returns:
        tf tensor of shape (..., m, m), a matrix without diagonal and upper
        triangular parts"""

    dim = X.shape[-1]
    dtype = X.dtype
    lower = tf.ones((dim, dim), dtype=dtype) - tf.linalg.diag(tf.ones((dim,),
                                                              dtype))
    lower = tf.linalg.band_part(lower, -1, 0)

    return lower * X


def _half(X):
    """Returns the lower triangular part of a matrix with half of diagonal part.
    Args:
        X: tf tensor of shape (..., m, m)
    Returns:
        tf tensor of shape (..., m, m), a matrix with half of diagonal and
        without upper triangular parts"""

    dim = X.shape[-1]
    dtype = X.dtype
    half = tf.ones((dim, dim),
                   dtype=dtype) - 0.5 * tf.linalg.diag(tf.ones((dim,), dtype))
    half = tf.linalg.band_part(half, -1, 0)

    return half * X


def _pull_back_chol(W, L, inv_L):
    """Takes a tangent vector to a point from S++ and
    computes the corresponding tangent vector to the
    corresponding point from L+
    Args:
        W: tf tensor of shape (..., m, m), tangent vector
        L: tf tensor of shape (..., m, m), triangular matrix from L+
        inv_L: tf tensor of shape (..., m, m), inverse L
    Returns:
        tf tensor of shape (..., m, m), tangent vector to corresponding
        point in L+"""

    X = inv_L @ W @ adj(inv_L)
    X = L @ (_half(X))

    return X


def _push_forward_chol(X, L):
    """Takes a tangent vector to a point from L+ and
    computes the corresponding tangent vector to the
    corresponding point from S++
    Args:
        X: tf tensor of shape (..., m, m), tangent vector
        L: tf tensor of shape (..., m, m), triangular matrix from L+
    Returns:
        tf tensor of shape (..., m, m), tangent vector to corresponding
        point in S++"""
    return L @ adj(X) + X @ adj(L)


def _f_matrix(lmbd):
    """Returns f matrix (part of pull back and push forward)
    Args:
        lmbd: tf tensor of shape (..., m), eigenvalues of matrix
        from S
    Returns:
        tf tensor of shape (..., m, m), f matrix"""

    n = lmbd.shape[-1]
    l_i = lmbd[..., tf.newaxis]
    l_j = lmbd[..., tf.newaxis, :]
    denom = tf.math.log(l_i / l_j) + tf.eye(n, dtype=lmbd.dtype)
    return (l_i - l_j) / denom + tf.linalg.diag(lmbd)


def _pull_back_log(W, U, lmbd):
    """Takes a tangent vector to a point from S++ and
    computes the corresponding tangent vector to the
    corresponding point from S
    Args:
        W: tf tensor of shape (..., m, m), tangent vector
        U: tf tensor of shape (..., m, m), unitary matrix
        from eigen decomposition of a point from S++
        lmbd: tf tensor of shape (..., m), eigenvalues of
        a point from S++
    Returns:
        tf tensor of shape (..., m, m), tangent vector to corresponding
        point in S"""

    f = _f_matrix(lmbd)

    return U @ ((1 / f) * (adj(U) @ W @ U)) @ adj(U)


def _push_forward_log(W, U, lmbd):
    """Takes a tangent vector to a point from S and
    computes the corresponding tangent vector to the
    corresponding point from S++
    Args:
        W: tf tensor of shape (..., m, m), tangent vector
        U: tf tensor of shape (..., m, m), unitary matrix
        from eigen decomposition of a point from S
        lmbd: tf tensor of shape (..., m), eigenvalues of
        a point from S
    Returns:
        tf tensor of shape (..., m, m), tangent vector to corresponding
        point in S++"""
    f = _f_matrix(lmbd)

    return U @ (f * (adj(U) @ W @ U)) @ adj(U)


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
            vec: complex valued tf.Tensor of shape (..., q, q), a direction vector
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
