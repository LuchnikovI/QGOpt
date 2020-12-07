import tensorflow as tf


def adj(A):
    """Batch conjugate transpose.

    Args:
        A: complex valued tensor of shape (..., n, m).

    Returns:
        complex valued tensor of shape (..., m, n),
        conjugate transpose matrix."""

    return tf.math.conj(tf.linalg.matrix_transpose(A))


def _lower(X):
    """Returns the lower triangular part of a matrix without the
    diagonal part.

    Args:
        X: tensor of shape (..., m, m).

    Returns:
        tensor of shape (..., m, m), a set of matrices without
        diagonal and upper triangular parts."""

    dim = tf.shape(X)[-1]
    dtype = X.dtype
    lower = tf.ones((dim, dim), dtype=dtype) - tf.linalg.diag(tf.ones((dim,),
                                                              dtype))
    lower = tf.linalg.band_part(lower, -1, 0)

    return lower * X


def _half(X):
    """Returns the lower triangular part of
    a matrix with half of diagonal part.

    Args:
        X: tensor of shape (..., m, m).

    Returns:
        tensor of shape (..., m, m), a set of matrices with half
        of diagonal and without upper triangular parts."""

    dim = tf.shape(X)[-1]
    dtype = X.dtype
    half = tf.ones((dim, dim),
                   dtype=dtype) - 0.5 * tf.linalg.diag(tf.ones((dim,), dtype))
    half = tf.linalg.band_part(half, -1, 0)

    return half * X


def _pull_back_chol(W, L, inv_L):
    """Takes a tangent vector to a point from S++ and
    computes the corresponding tangent vector to the
    corresponding point from L+.

    Args:
        W: complex valued tensor of shape (..., m, m), tangent vector.
        L: complex valued tensor of shape (..., m, m), 
            triangular matrix from L+.
        inv_L: complex valued tensor of shape (..., m, m), inverse L.

    Returns:
        complex valued tensor of shape (..., m, m), tangent vector
        to corresponding point in L+."""

    X = inv_L @ W @ adj(inv_L)
    X = L @ _half(X)

    return X


def _push_forward_chol(X, L):
    """Takes a tangent vector to a point from L+ and
    computes the corresponding tangent vector to the
    corresponding point from S++.

    Args:
        X: complex valued tensor of shape (..., m, m), tangent vector.
        L: complex valued tensor of shape (..., m, m),
            triangular matrix from L+.

    Returns:
        complex valued tensor of shape (..., m, m), tangent vector
        to corresponding point in S++."""

    return L @ adj(X) + X @ adj(L)


def _f_matrix(lmbd):
    """Returns f matrix (an auxiliary matrix for _pull_back_log
    and _push_forward_log).

    Args:
        lmbd: tensor of shape (..., m), eigenvalues of matrix
        from S.

    Returns:
        tensor of shape (..., m, m), f matrix."""

    n = tf.shape(lmbd)[-1]
    l_i = lmbd[..., tf.newaxis]
    l_j = lmbd[..., tf.newaxis, :]
    denom = tf.math.log(l_i / l_j) + tf.eye(n, dtype=lmbd.dtype)
    return (l_i - l_j) / denom + tf.linalg.diag(lmbd)


def _pull_back_log(W, U, lmbd):
    """Takes a tangent vector to a point from S++ and
    computes the corresponding tangent vector to the
    corresponding point from S.

    Args:
        W: complex valued tensor of shape (..., m, m),
            tangent vector.
        U: complex valued tensor of shape (..., m, m),
            unitary matrix from eigen decomposition of 
            a point from S++.
        lmbd: tensor of shape (..., m), eigenvalues of
            a point from S++.

    Returns:
        complex valued tensor of shape (..., m, m),
        tangent vector to the corresponding point in S"""

    f = _f_matrix(lmbd)
    return U @ ((1 / f) * (adj(U) @ W @ U)) @ adj(U)


def _push_forward_log(W, U, lmbd):
    """Takes a tangent vector to a point from S and
    computes the corresponding tangent vector to the
    corresponding point from S++.

    Args:
        W: complex valued tensor of shape (..., m, m),
            tangent vector.
        U: complex valued tensor of shape (..., m, m),
            unitary matrix from eigen decomposition of
            a point from S.
        lmbd: tensor of shape (..., m), eigenvalues of
            a point from S.

    Returns:
        complex valued tensor of shape (..., m, m), tangent
        vector to corresponding point in S++."""

    f = _f_matrix(lmbd)
    return U @ (f * (adj(U) @ W @ U)) @ adj(U)


def lyap_symmetric(A, C, eps=1e-9):
    """Solves AX + XA = C when A = adj(A).

    Args:
        A: complex valued tensor of shape (..., m, m).
        C: complex valued tensor of shape (..., m, m).
        eps: small float number guarantees safe inverse.

    Return:
        complex valued tensor of shape (..., m, m),
        solution of the equation.

    Note:
        The complexity O(m^3)"""

    lmbd, u = tf.linalg.eigh(A)
    uCu = adj(u) @ C @ u
    L = lmbd[..., tf.newaxis, :] + lmbd[..., tf.newaxis]
    return u @ (uCu * L / (L ** 2 + eps ** 2)) @ adj(u)


def shape_conc(*shapes):
    """Concatenates two shapes into one.
    
    Args:
        one-dimensional int valued tensors representing
        shapes.

    Return:
        one dimensional int tensor, concatenated shape."""
        
    return tf.concat(shapes, axis=0)
