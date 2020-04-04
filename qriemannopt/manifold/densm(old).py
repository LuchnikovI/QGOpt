import tensorflow as tf
from qriemannopt.manifold import base_manifold

def adj(A):
    """Correct adjoint
    Args:
        A: tf.tensor of shape (..., n, m)
    Returns:
        tf tensor of shape (..., m, n), adjoint matrix"""

    return tf.math.conj(tf.linalg.matrix_transpose(A))

def lower(X):
    """Returns lower triangular part of matrix without diagonal part.
    Args:
        X: tf tensor of shape (dim, dim)
    Returns:
        tf tensor of shape (dimm, dim), matrix without diagonal and upper
        triangular parts"""

    dim = X.shape[-1]
    dtype = X.dtype
    lower = tf.ones((dim, dim), dtype=dtype) - tf.linalg.diag(tf.ones((dim,), dtype))
    lower = tf.linalg.band_part(lower, -1, 0)

    return lower * X

def half(X):
    """Returns lower triangular part of matrix with half of diagonal part.
    Args:
        X: tf tensor of shape (dim, dim)
    Returns:
        tf tensor of shape (dimm, dim), matrix with half of diagonal and
        without upper triangular parts"""
        
    dim = X.shape[-1]
    dtype = X.dtype
    half = tf.ones((dim, dim), dtype=dtype) -\
                    0.5 * tf.linalg.diag(tf.ones((dim,), dtype))
    half = tf.linalg.band_part(half, -1, 0)

    return half * X


def pull_back_tangent(W, L, inv_L):
    """W is a vector from tangent space to S++.
    L is cholesky decomposition of a point in S++."""

    X = inv_L @ W @ adj(inv_L)
    X = L @ (half(X))

    return X

def push_forward_tangent(X, L):

    return L @ adj(X) + X @ adj(L)

'''def pinv(X):
    
    s, u, v = tf.linalg.svd(X)
    s_inv = tf.cast(s > 1e-8, dtype=s.dtype) / s
    s_inv = tf.cast(s_inv, dtype=u.dtype)
    return tf.einsum('ij,j,jk->ik', v, s_inv, adj(u))'''
    
def pinv(X):
    
    return tf.linalg.inv(X)
    

class DensM(base_manifold.Manifold):
    """Class is used to work with manifold of density matrices.
    It allows performing all
    necessary operations with elements of manifolds direct product and
    tangent spaces for optimization."""
    # TODO check correctness of transport for canonical metric
    
    
    def __init__(self):
        """Returns object of class DensM."""
        
        
    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        tangent space.
        Args:
            u: complex valued tensor of shape (..., q, q),
            element of manifolds direct product
            vec1: complex valued tensor of shape (..., q, q),
            vector from tangent space.
            vec2: complex valued tensor of shape (..., q, q),
            vector from tangent spaces.
        Returns:
            complex valued tensor of shape (...,),
            manifold wise inner product"""
            
        L = tf.linalg.cholesky(u)
        inv_L = pinv(L)

        X = pull_back_tangent(vec1, L, inv_L)
        Y = pull_back_tangent(vec2, L, inv_L)
        
        diag_inner = tf.math.conj(tf.linalg.diag_part(X)) *\
            tf.linalg.diag_part(Y) / (tf.linalg.diag_part(L) ** 2)
        diag_inner = tf.reduce_sum(diag_inner, axis=-1)
        triag_inner = tf.reduce_sum(tf.math.conj(lower(X))\
                        * lower(Y), axis=(-2, -1))
        
        return diag_inner + triag_inner
    
    
    def proj(self, u, vec):
        """Returns projection of vector on tangen space
        of direct product of manifolds.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            point of direct product.
            vec: complex valued tf.Tensor of shape (..., q, q),
            vectors to be projected.
        Returns:
            complex valued tf.Tensor of shape (..., q, q), projected vector"""
        n = u.shape[-1]
        return (vec + adj(vec)) / 2 - \
        tf.linalg.trace(vec)[..., tf.newaxis, tf.newaxis] *\
        tf.eye(n, dtype=u.dtype) / n
        

    def egrad_to_rgrad(self, u, egrad):
        """Returns riemannian gradient from euclidean gradient.
        Args:
            u: complex valued tf.Tensor of shape (..., q, q),
            element of direct product.
            egrad: complex valued tf.Tensor of shape (..., q, q),
            euclidean gradient.
        Returns:
            tf.Tensor of shape (..., q, q), reimannian gradient."""
            
        n = u.shape[-1]
        L = tf.linalg.cholesky(u)
        
        dtype = L.dtype
        half = tf.ones((n, n), dtype=dtype) - tf.linalg.diag(tf.ones((n,), dtype))
        G = tf.linalg.band_part(half, -1, 0) + tf.linalg.diag(tf.linalg.diag_part(L) ** 2)
        
        R = L @ adj(G * (egrad @ L))
        R = 2 * (R + adj(R))
        
        alpha = - tf.linalg.trace(R) / (tf.linalg.trace(G * u + adj(G * u)))
        
        R_corrected = R + alpha * (G * u + adj(G * u))
        
        return R_corrected
        
        #symgrad = (egrad + adj(egrad)) / 2
        #return u @ (symgrad - tf.eye(n, dtype=u.dtype) *\
        #            (tf.linalg.trace(u @ symgrad @ u) /\
        #            tf.linalg.trace(u @ u))[..., tf.newaxis, tf.newaxis]) @ u
             
    
    def retraction(self, u, vec):
        """Transports point via retraction map.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p), point
            to be transported
            vec: complex valued tf.Tensor of shape (..., q, p), vector of 
            direction
        Returns tf.Tensor of shape (..., q, p) new point"""

        L = tf.linalg.cholesky(u)
        inv_L = pinv(L)

        X = pull_back_tangent(vec, L, inv_L)

        #inv_diag_L = tf.linalg.diag(tf.linalg.diag_part(L) / (tf.linalg.diag_part(L) ** 2 + 1e-8))
        inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))
        #inv_diag_L = pinv(tf.linalg.band_part(L, 0, 0))
        
        cholesky_retraction = lower(L) + lower(X) + tf.linalg.band_part(L, 0, 0) * tf.exp(tf.linalg.band_part(X, 0, 0) * inv_diag_L)
        densm_retraction = cholesky_retraction @ adj(cholesky_retraction)
        
        #densm_retraction /= tf.linalg.trace(densm_retraction)[..., tf.newaxis, tf.newaxis]
        #densm_retraction += 1e-10 * tf.eye(densm_retraction.shape[-1], dtype = densm_retraction.dtype)
        
        return densm_retraction / tf.linalg.trace(densm_retraction)[..., tf.newaxis, tf.newaxis]
        
        
    def vector_transport(self, u, vec1, vec2):
        """Returns vector vec1 tranported from point u along vec2.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            initial point of direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            direction vector.
        Returns:
            complex valued tf.Tensor of shape (..., q, p),
            transported vector."""
            
        '''new_u = self.retraction(u, vec2)
        new_u_evals, new_u_evec = tf.linalg.eigh(new_u)
        u_evals, u_evec = tf.linalg.eigh(u)
        sqrt_u_new = (new_u_evec *\
                      tf.math.sqrt(new_u_evals)[..., tf.newaxis, :]) @\
                      tf.linalg.adjoint(new_u_evec)
        sqrt_u_inv = (u_evec *\
                      1 / tf.math.sqrt(u_evals)[..., tf.newaxis, :]) @\
                      tf.linalg.adjoint(u_evec)'''
        v = self.retraction(u, vec2)

        L = tf.linalg.cholesky(u)
        inv_L = pinv(L)
        #inv_diag_L = tf.linalg.diag(tf.linalg.diag_part(L) / (tf.linalg.diag_part(L) ** 2 + 1e-8))
        inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))
        #inv_diag_L = pinv(tf.linalg.band_part(L, 0, 0))
        
        X = pull_back_tangent(vec1, L, inv_L)

        K = tf.linalg.cholesky(v)
        
        L_transport = lower(X) + tf.linalg.band_part(K, 0, 0) * inv_diag_L * tf.linalg.band_part(X, 0, 0)

        transport = K @ adj(L_transport) + L_transport @ adj(K)
        
        return self.proj(v, transport)
    
    
    def retraction_transport(self, u, vec1, vec2):
        """Performs retraction and vector transport at the same time.
        Args:
            u: complex valued tf.Tensor of shape (..., q, p),
            initial point from direct product.
            vec1: complex valued tf.Tensor of shape (..., q, p),
            vector to be transported.
            vec2: complex valued tf.Tensor of shape (..., q, p),
            direction vector.
        Returns:
            two complex valued tf.Tensor of shape (..., q, p),
            new point and transported vector."""
        
        v = self.retraction(u, vec2)

        L = tf.linalg.cholesky(u)
        inv_L = pinv(L)
        #inv_diag_L = tf.linalg.diag(tf.linalg.diag_part(L) / (tf.linalg.diag_part(L) ** 2 + 1e-8))
        inv_diag_L = tf.linalg.diag(1 / tf.linalg.diag_part(L))
        #inv_diag_L = pinv(tf.linalg.band_part(L, 0, 0))

        X = pull_back_tangent(vec1, L, inv_L)

        K = tf.linalg.cholesky(v)
        
        L_transport = lower(X) + tf.linalg.band_part(K, 0, 0) * inv_diag_L * tf.linalg.band_part(X, 0, 0)

        transport = K @ adj(L_transport) + L_transport @ adj(K)
        
        return v, self.proj(v, transport)
