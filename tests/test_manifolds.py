import QGOpt.manifolds as manifolds
import pytest
import tensorflow as tf

def rank3_norm(A):
    """Returns norm of a tesnor wrt 3 last indices.

    Args:
        A: tensor of rank >=3

    Return:
        norm of a tesnor wrt 3 last indices"""
    return tf.math.sqrt(tf.reduce_sum(A * tf.math.conj(A), axis=(-3, -2, -1)))
    

class CheckManifolds():

    def __init__(self, m, descr, shape, tol):

        self.m = m  # example of a manifold
        self.descr = descr
        self.shape = shape  # shape of a tensor
        self.u = m.random(shape, dtype=tf.complex128)  # point from a manifold
        self.v1 = m.random_tangent(self.u)  # first tangent vector
        self.v2 = m.random_tangent(self.u)  # second tangent vector
        self.zero = self.u * 0.  # zero vector
        self.tol = tol  # tolerance of a test

    def _proj_of_tangent(self):
        """
        Checking m.proj: Projection of a tangent vector should remain the same
        after application of the proj method.

        Args:

        Returns:
            tf scalar, maximum value of error
        """

        rank = self.m.rank
        if rank == 2:
            err = tf.linalg.norm(self.v1 - self.m.proj(self.u, self.v1),
                                                        axis=(-2, -1))
        elif rank == 3:
            err = rank3_norm(self.v1 - self.m.proj(self.u, self.v1))
        err = tf.reduce_max(tf.math.real(err))
        return err

    def _inner_proj_matching(self):
        """
        Checking matching between m.inner and m.proj
        (suitable only for embedded manifolds with globally defined metric)
        Args:

        Returns:
            tf scalar, maximum value of error
        """

        xi = tf.complex(tf.random.normal(self.shape),
                        tf.random.normal(self.shape))
        xi = tf.cast(xi, dtype=self.u.dtype)

        xi_proj = self.m.proj(self.u, xi)
        first_inner = self.m.inner(self.u, xi_proj, self.v1)
        second_inner = self.m.inner(self.u, xi, self.v1)
        err = tf.abs(first_inner - second_inner)
        if self.m.rank == 2:
            err = err[..., 0, 0]
        elif self.m.rank == 3:
            err = err[..., 0, 0, 0]
        err = tf.reduce_max(err)
        return err

    def _retraction(self):
        """
        Checking retraction
        Page 46, Nikolas Boumal, An introduction to optimization on smooth
        manifolds.
        1) Rx(0) = x (Identity mapping)
        2) DRx(o)[v] = v : introduce v->t*v, calculate err=dRx(tv)/dt|_{t=0}-v
        3) Presence of a new point in a manifold

        Args:

        Returns:
            list with three tf scalars. First two scalars give maximum
            violation of first two conditions, third scalar shows wether
            any point

        """

        dt = 1e-8  # dt for numerical derivative

        # transition along zero vector (first cond)
        err1 = self.u - self.m.retraction(self.u, self.zero)
        if self.m.rank == 2:
            err1 = tf.math.real(tf.linalg.norm(err1, axis=(-2, -1)))
        if self.m.rank == 3:
            err1 = tf.math.real(rank3_norm(err1))

        # third order approximation of differential of retraction (second cond)
        t = tf.constant(dt, dtype=self.u.dtype)
        retr_forward = self.m.retraction(self.u, t * self.v1)
        retr_forward_two_steps = self.m.retraction(self.u, 2 * t * self.v1)
        retr_back = self.m.retraction(self.u, -t * self.v1)
        dretr = (-2 * retr_back - 3 * self.u + 6 * retr_forward - retr_forward_two_steps)
        dretr = dretr / (6 * t)
        if self.m.rank == 2:
            err2 = tf.math.real(tf.linalg.norm(dretr - self.v1,
                                               axis=(-2, -1)))
        elif self.m.rank == 3:
            err2 = tf.math.real(rank3_norm(dretr - self.v1))

        # presence of a new point in a manifold (third cond)
        err3 = self.m.is_in_manifold(self.m.retraction(self.u, self.v1),
                                                                tol=self.tol)
        err1 = tf.reduce_max(err1)
        err2 = tf.reduce_max(err2)
        err3 = tf.reduce_any(err3)
        return err1, err2, err3

    def _vector_transport(self):
        """
        Checking vector transport.
        Page 264, Nikolas Boumal, An introduction to optimization on smooth
        manifolds.
        1) transported vector lies in a new tangent space
        2) VT(x,0)[v] is the identity mapping on TxM.

        Args:

        Returns:
            list with two tf scalars that give maximum
            violation of two conditions
        """

        vt = self.m.vector_transport(self.u, self.v1, self.v2)
        err1 = vt - self.m.proj(self.m.retraction(self.u, self.v2), vt)
        if self.m.rank == 2:
            err1 = tf.math.real(tf.linalg.norm(err1, axis=(-2, -1)))
        elif self.m.rank == 3:
            err1 = tf.math.real(rank3_norm(err1))

        err2 = self.v1 - self.m.vector_transport(self.u, self.v1, self.zero)
        if self.m.rank == 2:
            err2 = tf.math.real(tf.linalg.norm(err2, axis=(-2, -1)))
        elif self.m.rank == 3:
            err2 = tf.math.real(rank3_norm(err2))
        err1 = tf.reduce_max(err1)
        err2 = tf.reduce_max(err2)
        return err1, err2

    def _egrad_to_rgrad(self):
        """
        Checking egrad_to_rgrad method.
        1) rgrad is in the tangent space of a manifold's point
        2) <v1 egrad> = <v1 rgrad>_m (matching between egrad and rgrad)
        Args:

        Returns:
            list with two tf scalars that give maximum
            violation of two conditions
        """

        # vector that plays the role of a gradient
        xi = tf.random.normal(self.u.shape + (2,))
        xi = manifolds.real_to_complex(xi)
        xi = tf.cast(xi, dtype=self.u.dtype)

        # rgrad
        rgrad = self.m.egrad_to_rgrad(self.u, xi)

        err1 = rgrad - self.m.proj(self.u, rgrad)
        if self.m.rank == 2:
            err1 = tf.math.real(tf.linalg.norm(err1, axis=(-2, -1)))
        elif self.m.rank == 3:
            err1 = tf.math.real(rank3_norm(err1))

        if self.m.rank == 2:
            err2 = tf.reduce_sum(tf.math.conj(self.v1) * xi, axis=(-2, -1)) -\
                            self.m.inner(self.u, self.v1, rgrad)[..., 0, 0]
        elif self.m.rank == 3:
            err2 = tf.reduce_sum(tf.math.conj(self.v1) * xi, axis=(-3, -2, -1)) -\
                            self.m.inner(self.u, self.v1, rgrad)[..., 0, 0, 0]

        err2 = tf.abs(tf.math.real(err2))

        err1 = tf.reduce_max(err1)
        err2 = tf.reduce_max(err2)
        return err1, err2

    def checks(self):
        # TODO after checking: rewrite with asserts
        """
        Routine for pytest: checking tolerance of manifold functions
        """
        err = self._proj_of_tangent()
        assert err < self.tol, "Projection error for:{}.\
                    ".format(self.descr)
        # no need to test .proj of log_cholesky, it does not
        # take part in any optimization algorithms
        if self.descr[1] not in ['log_cholesky']:
            err = self._inner_proj_matching()
            assert err < self.tol, "Inner/proj error for:{}.\
                    ".format(self.descr)

        err1, err2, err3 = self._retraction()
        assert err1 < self.tol, "Retraction (Rx(0) != x) error for:{}.\
                    ".format(self.descr)
        assert err2 < self.tol, "Retraction (DRx(o)[v] != v) error for:{}.\
                    ".format(self.descr)
        assert err3 == True, "Retraction (not in manifold) error for:{}.\
                    ".format(self.descr)

        err1, err2 = self._vector_transport()
        assert err1 < self.tol, "Vector transport (not in a TMx) error for:{}.\
                    ".format(self.descr)
        assert err2 < self.tol, "Vector transport (VT(x,0)[v] != v) error for:\
                    {}.".format(self.descr)

        err1, err2 = self._egrad_to_rgrad()
        #TODO separate rgrad test for quotient manifolds
        if self.descr[0] not in ['ChoiMatrix', 'DensityMatrix', 'POVM']:
            assert err1 < self.tol, "Rgrad (not in a TMx) error for:{}.\
                    ".format(self.descr)
        assert err2 < self.tol, "Rgrad (<v1 egrad> != inner<v1 rgrad>) \
                    error for:{}.".format(self.descr)

#TODO find a problem with tests or/and PositiveCone manifold
testdata = [
    ('ChoiMatrix', 'euclidean', manifolds.ChoiMatrix(metric='euclidean'), (4, 4), 1.e-6),
    ('DensityMatrix', 'euclidean', manifolds.DensityMatrix(metric='euclidean'), (4, 4), 1.e-6),
    ('HermitianMatrix', 'euclidean', manifolds.HermitianMatrix(metric='euclidean'), (4, 4), 1.e-6),
    ('PositiveCone', 'log_euclidean', manifolds.PositiveCone(metric='log_euclidean'), (4, 4), 1.e-5),
    ('PositiveCone', 'log_cholesky', manifolds.PositiveCone(metric='log_cholesky'), (4, 4), 1.e-5),
    ('POVM', 'euclidean', manifolds.POVM(metric='euclidean'), (4, 2, 2), 1.e-6)
]

@pytest.mark.parametrize("name,metric,manifold,shape,tol", testdata)
def test_manifolds(name, metric, manifold, shape, tol):
    Test = CheckManifolds(manifold, (name, metric), shape, tol)
    Test.checks()
