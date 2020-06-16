from QGOpt.manifolds import base_manifold
import tensorflow as tf
from QGOpt.manifolds.utils import adj
from QGOpt.manifolds.utils import lyap_symmetric
import math


class ChoiMatrix(base_manifold.Manifold):
    """The manifold of Choi matrices of fixed rank (Kraus rank).
    Choi matrices of fixed Kraus rank are the set of matrices of
    size (n^2)x(n^2) (n is the dimension of a quantum system) with rank k
    that are positive definite (the corresponding quantum channel is
    completely positive) and with the additional constraint Tr_2(choi) = Id,
    where Tr_2 is the partial trace over the second subsystem, Id is
    the identity matrix (ensures the trace-preserving property of the
    corresponding quantum channel). In the general case Kraus rank of
    a Choi matrix is equal to n^2. An element of this manifold is
    represented by a complex matrix A of size (n^2)xk that parametrizes
    a Choi matrix choi = A @ adj(A) (positive by construction).
    Notice that for any unitary matrix Q of size kxk the transformation
    A --> AQ leaves resulting matrix the same. This fact is taken into
    account by consideration of quotient manifold from

    Yatawatta, S. (2013, May). Radio interferometric calibration using a
    Riemannian manifold. In 2013 IEEE International Conference on Acoustics,
    Speech and Signal Processing (pp. 3866-3870). IEEE.

    Args:
        metric: string specifies type of metric, Defaults to 'euclidean'.
            Types of metrics are available: 'euclidean'.

    Notes:
        All methods of this class operates with tensors of shape (..., n ** 2, k),
        where (...) enumerates manifold (can be any shaped), (n ** 2, k)
        is the shape of a particular matrix (e.g. an element of the manifold
        or its tangent vector).
        In order to take a partial trace of a choi matrix over the second
        subsystem one can at first reshape a choi matrix
        (n ** 2, n ** 2) --> (n, n, n, n) and then take a trace over
        1st and 3rd indices (the numeration starts from 0)."""

    def __init__(self, metric='euclidean'):

        list_of_metrics = ['euclidean']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")

    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k),
                a set of points from the manifold.
            vec1: complex valued tensor of shape (..., n ** 2, k),
                a set of tangent vectors from the manifold.
            vec2: complex valued tensor of shape (..., n ** 2, k),
                a set of tangent vectors from the manifold.

        Returns:
            complex valued tensor of shape (..., 1, 1),
                manifold wise inner product"""

        prod = tf.reduce_sum(tf.math.conj(vec1) * vec2, axis=(-2, -1))
        prod = tf.math.real(prod)
        prod = tf.cast(prod, dtype=u.dtype)[..., tf.newaxis, tf.newaxis]
        return prod

    def proj(self, u, vec):
        """Returns projection of vectors on a tangen space
        of the manifold.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k),
                a set of points from the manifold.
            vec: complex valued tensor of shape (..., n ** 2, k),
                a set of vectors to be projected.

        Returns:
            complex valued tensor of shape (..., n ** 2, k),
            a set of projected vectors."""

        k = u.shape[-1]
        n = int(math.sqrt(u.shape[-2]))
        shape = u.shape[:-2]

        # projection onto the tangent space of the Stiefel manifold
        vec_mod = tf.reshape(vec, shape + (n, k * n))
        vec_mod = tf.linalg.matrix_transpose(vec_mod)

        u_mod = tf.reshape(u, shape + (n, k * n))
        u_mod = tf.linalg.matrix_transpose(u_mod)
        vec_mod = vec_mod - 0.5 * u_mod @ (adj(u_mod) @ vec_mod +\
                                           adj(vec_mod) @ u_mod)
        vec_mod = tf.linalg.matrix_transpose(vec_mod)
        vec_mod = tf.reshape(vec_mod, shape + (n ** 2, k))

        # projection onto the horizontal space
        uu = adj(u) @ u
        Omega = lyap_symmetric(uu, adj(u) @ vec_mod - adj(vec_mod) @ u)
        return vec_mod - u @ Omega

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k),
                a set of points from the manifold.
            egrad: complex valued tensor of shape (..., n ** 2, k),
                a set of Euclidean gradients.

        Returns:
            complex valued tensor of shape (..., n ** 2, k),
            the set of Reimannian gradients."""

        k = u.shape[-1]
        n = int(math.sqrt(u.shape[-2]))
        shape = u.shape[:-2]

        # projection onto the tangent space of the Stiefel manifold
        vec_mod = tf.reshape(egrad, shape + (n, k * n))
        vec_mod = tf.linalg.matrix_transpose(vec_mod)

        u_mod = tf.reshape(u, shape + (n, k * n))
        u_mod = tf.linalg.matrix_transpose(u_mod)
        vec_mod = vec_mod - 0.5 * u_mod @ (adj(u_mod) @ vec_mod +\
                                           adj(vec_mod) @ u_mod)
        vec_mod = tf.linalg.matrix_transpose(vec_mod)
        vec_mod = tf.reshape(vec_mod, shape + (n ** 2, n ** 2))
        return vec_mod

    def retraction(self, u, vec):
        """Transports a set of points from the manifold via a
        retraction map.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k), a set
                of points to be transported.
            vec: complex valued tensor of shape (..., n ** 2, k),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n ** 2, k),
            a set of transported points."""

        k = u.shape[-1]
        n = int(math.sqrt(u.shape[-2]))
        shape = u.shape[:-2]

        # svd based retraction
        u_new = u + vec
        u_new = tf.reshape(u_new, shape + (n, n * k))
        _, U, V = tf.linalg.svd(u_new)

        u_new = U @ adj(V)
        u_new = tf.reshape(u_new, shape + (n ** 2, n ** 2))
        return u_new

    def vector_transport(self, u, vec1, vec2):
        """Returns a vector tranported along an another vector
        via vector transport.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k),
                a set of points from the manifold, starting points.
            vec1: complex valued tensor of shape (..., n ** 2, k),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n ** 2, k),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n ** 2, k),
            a set of transported vectors."""

        u_new = self.retraction(u, vec2)
        return self.proj(u_new, vec1)

    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport simultaneously.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k),
                a set of points from the manifold, starting points.
            vec1: complex valued tensor of shape (..., n ** 2, k),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n ** 2, k),
                a set of direction vectors.

        Returns:
            two complex valued tensors of shape (..., n ** 2, k),
            a set of transported points and vectors."""

        u_new = self.retraction(u, vec2)
        return u_new, self.proj(u_new, vec1)

    def random(self, shape, dtype=tf.complex64):
        """Returns a set of points from the manifold generated
        randomly.

        Args:
            shape: tuple of integer numbers (..., n ** 2, k),
                shape of a generated matrix.
            dtype: type of an output tensor, can be
                either tf.complex64 or tf.complex128.

        Returns:
            complex valued tensor of shape (..., n ** 2, k),
            a generated matrix."""

        list_of_dtypes = [tf.complex64, tf.complex128]

        if dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")
        real_dtype = tf.float64 if dtype == tf.complex128 else tf.float32

        u = tf.complex(tf.random.normal(shape, dtype=real_dtype),
                       tf.random.normal(shape, dtype=real_dtype))

        k = shape[-1]
        n = int(math.sqrt(shape[-2]))

        u = tf.reshape(u, shape[:-2] + (n, n * k))
        u = tf.linalg.matrix_transpose(u)
        u, _ = tf.linalg.qr(u)
        u = tf.linalg.matrix_transpose(u)
        u = tf.reshape(u, shape[:-2] + (n ** 2, k))

        return u

    def random_tangent(self, u):
        """Returns a set of random tangent vectors to points from
        the manifold.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k), points
                from the manifold.

        Returns:
            complex valued tensor, set of tangent vectors to u."""

        vec = tf.complex(tf.random.normal(u.shape), tf.random.normal(u.shape))
        vec = tf.cast(vec, dtype=u.dtype)
        vec = self.proj(u, vec)
        return vec

    def is_in_manifold(self, u, tol=1e-5):
        """Checks if a point is in the manifold or not.

        Args:
            u: complex valued tensor of shape (..., n ** 2, k),
                a point to be checked.
            tol: small real value showing tolerance.

        Returns:
            bolean tensor of shape (...)."""

        shape = u.shape[:-2]
        n_sq, k = u.shape[-2], u.shape[-1]
        n = int(tf.math.sqrt(tf.cast(n_sq, dtype=tf.float32)))
        u_resh = tf.reshape(u, shape + (n, n * k))
        uudag = u_resh @ adj(u_resh)
        Id = tf.eye(uudag.shape[-1], dtype=u.dtype)
        diff = tf.linalg.norm(uudag - Id, axis=(-2, -1))
        uudag_norm = tf.linalg.norm(uudag, axis=(-2, -1))
        Id_norm = tf.linalg.norm(Id, axis=(-2, -1))
        rel_diff = tf.abs(diff / tf.math.sqrt(Id_norm * uudag_norm))
        return tol > rel_diff
