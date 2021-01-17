from QGOpt.manifolds import base_manifold
import tensorflow as tf
from QGOpt.manifolds.utils import adj
from QGOpt.manifolds.utils import lyap_symmetric


class DensityMatrix(base_manifold.Manifold):
    """The manifold of density matrices of fixed rank (rho(n, r) the positive
    definite hermitian matrices of size nxn with unit trace and rank r).
    An element of the manifold is represented by a complex matrix A that
    parametrizes density matrix rho = A @ adj(A)  (positive by construction).
    Notice that for any unitary matrix Q of size nxn the transformation A --> AQ
    leaves resulting matrix the same. This fact is taken into account by
    consideration of quotient manifold from

    Yatawatta, S. (2013, May). Radio interferometric calibration using a
    Riemannian manifold. In 2013 IEEE International Conference on Acoustics,
    Speech and Signal Processing (pp. 3866-3870). IEEE.

    It is also partly inspired by the Manopt package (www.manopt.org).

    Args:
        metric: string specifies type of metric, Defaults to 'euclidean'.
            Types of metrics are available: 'euclidean'.
        retraction: string specifies type of retraction, Defaults to 'projection'.
            Types of metrics are available: 'projection'.

    Notes:
        All methods of this class operates with tensors of shape (..., n, r),
        where (...) enumerates manifold (can be any shaped), (n, r)
        is the shape of a particular matrix (e.g. an element of the manifold
        or its tangent vector)."""

    def __init__(self, retraction='projection', metric='euclidean'):

        self.rank = 2
        self.quotient = True
        list_of_metrics = ['euclidean']
        list_of_retractions = ['projection']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")
        if retraction not in list_of_retractions:
            raise ValueError("Incorrect retraction")

        super(DensityMatrix, self).__init__(retraction, metric)

    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space.

        Args:
            u: complex valued tensor of shape (..., n, r),
                a set of points from the manifold.
            vec1: complex valued tensor of shape (..., n, r),
                a set of tangent vectors from the manifold.
            vec2: complex valued tensor of shape (..., n, r),
                a set of tangent vectors from the manifold.

        Returns:
            complex valued tensor of shape (..., 1, 1),
            manifold wise inner product.

        Note:
            The complexity is O(nr)."""

        prod = tf.reduce_sum(tf.math.conj(vec1) * vec2, axis=(-2, -1))
        prod = tf.math.real(prod)
        prod = tf.cast(prod, dtype=u.dtype)
        return prod[..., tf.newaxis, tf.newaxis]

    def proj(self, u, vec):
        """Returns projection of vectors on a tangen space
        of the manifold.

        Args:
            u: complex valued tensor of shape (..., n, r),
                a set of points from the manifold.
            vec: complex valued tensor of shape (..., n, r),
                a set of vectors to be projected.

        Returns:
            complex valued tensor of shape (..., n, r),
            a set of projected vectors.

        Note:
            The complexity is O(nr^2)."""

        # projection onto the tangent space of ||u||_F = 1
        vec_proj = vec - u * tf.reduce_sum(tf.math.conj(u) * vec, axis=(-2, -1))[...,
                                            tf.newaxis, tf.newaxis]

        # projection onto the horizontal space
        uu = adj(u) @ u
        Omega = lyap_symmetric(uu, adj(u) @ vec_proj - adj(vec_proj) @ u)
        return vec_proj - u @ Omega

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.

        Args:
            u: complex valued tensor of shape (..., n, r),
                a set of points from the manifold.
            egrad: complex valued tensor of shape (..., n, r),
                a set of Euclidean gradients.

        Returns:
            complex valued tensor of shape (..., n, r),
            the set of Reimannian gradients.

        Note:
            The complexity is O(nr)."""

        rgrad = egrad - u * tf.reduce_sum(tf.math.conj(u) * egrad, axis=(-2, -1))[...,
                                           tf.newaxis, tf.newaxis]
        return rgrad

    def retraction(self, u, vec):
        """Transports a set of points from the manifold via a
        retraction map.

        Args:
            u: complex valued tensor of shape (..., n, r), a set
                of points to be transported.
            vec: complex valued tensor of shape (..., n, r),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, r),
            a set of transported points.

        Note:
            The complexity is O(nr)."""

        u_new = (u + vec)
        u_new = u_new / tf.linalg.norm(u_new, axis=(-2, -1), keepdims=True)
        return u_new

    def vector_transport(self, u, vec1, vec2):
        """Returns a vector tranported along an another vector
        via vector transport.

        Args:
            u: complex valued tensor of shape (..., n, r),
                a set of points from the manifold, starting points.
            vec1: complex valued tensor of shape (..., n, r),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, r),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, r),
            a set of transported vectors.

        Note:
            The complexity is O(nr^2)."""

        u_new = (u + vec2)
        u_new = u_new / tf.linalg.norm(u_new, axis=(-2, -1), keepdims=True)
        return self.proj(u_new, vec1)

    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport simultaneously.

        Args:
            u: complex valued tensor of shape (..., n, r),
                a set of points from the manifold, starting points.
            vec1: complex valued tensor of shape (..., n, r),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, r),
                a set of direction vectors.

        Returns:
            two complex valued tensors of shape (..., n, r),
            a set of transported points and vectors."""

        u_new = (u + vec2)
        u_new = u_new / tf.linalg.norm(u_new, axis=(-2, -1), keepdims=True)
        return u_new, self.proj(u_new, vec1)

    def random(self, shape, dtype=tf.complex64):
        """Returns a set of points from the manifold generated
        randomly.

        Args:
            shape: tuple of integer numbers (..., n, r),
                shape of a generated matrix.
            dtype: type of an output tensor, can be
                either tf.complex64 or tf.complex128.

        Returns:
            complex valued tensor of shape (..., n, r),
            a generated matrix."""

        list_of_dtypes = [tf.complex64, tf.complex128]

        if dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")
        real_dtype = tf.float64 if dtype == tf.complex128 else tf.float32

        u = tf.complex(tf.random.normal(shape, dtype=real_dtype),
                       tf.random.normal(shape, dtype=real_dtype))
        u = u / tf.linalg.norm(u, axis=(-2, -1))[..., tf.newaxis, tf.newaxis]
        return u

    def random_tangent(self, u):
        """Returns a set of random tangent vectors to points from
        the manifold.

        Args:
            u: complex valued tensor of shape (..., n, r), points
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

        length = tf.linalg.norm(u, axis=(-2, -1))
        return tol > tf.abs(length - 1)
