import tensorflow as tf
from qriemannopt.manifold import base_manifold

def adj(A):
    """Correct adjoint
    Args:
        A: tf.tensor of shape (..., n, m)
    Returns:
        tf tensor of shape (..., m, n), adjoint matrix"""

    return tf.math.conj(tf.linalg.matrix_transpose(A))

def f_matrix(lmbd):
    
    n = lmbd.shape[-1]
    l_i = lmbd[..., tf.newaxis]
    l_j = lmbd[..., tf.newaxis, :]
    denom = tf.math.log(l_i) - tf.math.log(l_j) + tf.eye(n, dtype=lmbd.dtype)
    return (l_i - l_j) / denom + tf.linalg.diag(lmbd)

def pull_back(W, U, lmbd):
    """
    Takes tangent vector to S++ and computes tangent vector to S
    """
    f = f_matrix(lmbd)
    
    return U @ ((1/f) * (adj(U) @ W @ U)) @ adj(U)

def push_forward(W, U, lmbd):
    """
    Takes tangent vector to S and computes tangent vector to S++
    """
    f = f_matrix(lmbd)
    
    return U @ (f * (adj(U) @ W @ U)) @ adj(U)
    

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
        lmbd, U = tf.linalg.eigh(u)
        
        W = pull_back(vec1, U, lmbd)
        V = pull_back(vec2, U, lmbd)
        
        return tf.linalg.trace(adj(W) @ V)
        
    
    
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
        
        lmbd, U = tf.linalg.eigh(u)
        f = f_matrix(lmbd)
        
        E = adj(U) @ ((egrad + adj(egrad)) / 2) @ U
        
        R_temp = U @ (E * f * f) @ adj(U)
        alpha = - tf.linalg.trace(R_temp) / tf.linalg.trace(f * f)
        
        return R_temp + alpha * (U @ (f * f) @ adj(U))
        
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

        n = u.shape[-1]
        lmbd, U = tf.linalg.eigh(u)
        
        Su = U @ tf.linalg.diag(tf.math.log(lmbd)) @ adj(U)
        Svec = pull_back(vec, U, lmbd)
        
        Sresult = Su - Svec
        
        result = tf.linalg.expm(Sresult)
        
        return result / tf.linalg.trace(result)[..., tf.newaxis, tf.newaxis]
        
        
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
        n = u.shape[-1]
        lmbd, U = tf.linalg.eigh(u)
        
        Su = U @ tf.linalg.diag(tf.math.log(lmbd)) @ adj(U)
        Svec2 = pull_back(vec2, U, lmbd)
        
        Sresult = Su - Svec2
        
        transport = tf.linalg.expm(Sresult)
        
        alpha = tf.linalg.trace(transport)[..., tf.newaxis, tf.newaxis]
        
        new_point = transport / alpha
        
        new_lmbd = lmbd - tf.math.log(alpha)
        
        new_vec1 = push_forward(pull_back(vec1, U, lmbd), U, new_lmbd)

        return self.proj(new_vec1, new_point)
    
    
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
        
        n = u.shape[-1]
        lmbd, U = tf.linalg.eigh(u)
        
        Su = U @ tf.linalg.diag(tf.math.log(lmbd)) @ adj(U)
        Svec2 = pull_back(vec2, U, lmbd)
        
        Sresult = Su - Svec2
        
        transport = tf.linalg.expm(Sresult)
        
        alpha = tf.linalg.trace(transport)[..., tf.newaxis, tf.newaxis]
        
        new_point = transport / alpha
        
        new_lmbd = lmbd - tf.math.log(alpha)
        
        new_vec1 = push_forward(pull_back(vec1, U, lmbd), U, new_lmbd)
        
        return new_point, self.proj(new_vec1, new_point)
