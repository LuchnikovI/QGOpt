import QGOpt.manifolds as manifolds
import pytest
import tensorflow as tf


class CheckManifolds():
    def __init__(self, m, descr, shape, tol):
        self.m = m
        self.descr = descr
        self.shape = shape
        self.u = m.random(shape, dtype=tf.complex128)
        self.v1 = m.random_tangent(self.u)
        self.v2 = m.random_tangent(self.u)
        self.zero = self.u * 0.
        self.tol = tol

    def _proj_of_tangent(self):
        """
        Checking m.proj: Projection of a vector from a tangent space
        does't shange the vector
        Args:

        Returns:
            error, dtype float32
        """
        err = tf.linalg.norm(self.v1 - self.m.proj(self.u, self.v1),
                                                    axis=(-2, -1))
        return tf.math.real(err)

    def _inner_proj_matching(self):
        """
        Checking m.inner and m.proj
        arXiv:1906.10436v1: proj_ksi = argmin(-inner(z,ksi)+0.5inner(ksi,ksi))

        Args:

        Returns:
            error, number of False counts, dtype int
        """
        xi = tf.complex(tf.random.normal(self.shape),
                        tf.random.normal(self.shape))
        xi = tf.cast(xi, dtype=self.u.dtype)

        xi_proj = self.m.proj(self.u, xi)
        first_inner = self.m.inner(self.u, xi_proj, self.v1)
        second_inner = self.m.inner(self.u, xi, self.v1)
        err = tf.abs(first_inner - second_inner)
        return tf.cast(tf.math.real(err), dtype=tf.float32)

    def _retraction(self):
        """
        Checking retraction
        Page 46, Nikolas Boumal, An introduction to optimization on smooth
        manifolds
        1) Rx(0) = x (Identity mapping)
        2) DRx(o)[v] = v : introduce v->t*v, calculate err=dRx(tv)/dt|_{t=0}-v
        Args:

        Returns:
            error 1), dtype float32
            error 2), dtype float32
            error 3), dtype boolean
        """
        dt = 1e-8

        err1 = self.u - self.m.retraction(self.u, self.zero)
        err1 = tf.math.real(tf.linalg.norm(err1))

        t = tf.constant(dt, dtype=self.u.dtype)
        retr = self.m.retraction(self.u, t * self.v1)
        dretr = (retr - self.u) / dt
        err2 = tf.math.real(tf.linalg.norm(dretr - self.v1))

        err3 = self.m.is_in_manifold(self.m.retraction(self.u, self.v1),
                                                                tol=self.tol)
        return tf.cast(err1, dtype=tf.float32), tf.cast(err2,
                                            dtype=tf.float32), err3

    def _vector_transport(self):
        """
        Checking vector transport.
        Page 264, Nikolas Boumal, An introduction to optimization on smooth
        manifolds
        1) transported v2 in a tangent space
        2) VT(x,0)[v] is the identity on TxM.
        Args:

        Returns:
            error 1), dtype float32
            error 2), dtype float32
        """
        VT = self.m.vector_transport(self.u, self.v1, self.v2)
        err1 = VT - self.m.proj(self.m.retraction(self.u, self.v2), VT)
        err1 = tf.math.real(tf.linalg.norm(err1))

        err2 = self.v1 - self.m.vector_transport(self.u, self.v1, self.zero)
        err2 = tf.math.real(tf.linalg.norm(err2))
        return tf.cast(err1, dtype=tf.float32), tf.cast(err2, dtype=tf.float32)

    def _egrad_to_rgrad(self):
        """
        Checking egrad to rgrad.
        1) rgrad is in a tangent space
        2) <v1 egrad> = inner<v1 rgrad>
        Args:

        Returns:
            error 1), dtype float32
            error 2), dtype float32
        """

        xi = tf.random.normal(self.u.shape + (2,))
        xi = manifolds.real_to_complex(xi)
        xi = tf.cast(xi, dtype=self.u.dtype)
        rgrad = self.m.egrad_to_rgrad(self.u, xi)
        err1 = rgrad - self.m.proj(self.u, rgrad)
        err1 = tf.abs(tf.linalg.norm(err1))

        err2 = tf.reduce_sum(tf.math.conj(self.v1) * xi, axis=(-2, -1)) -\
                        self.m.inner(self.u, self.v1, rgrad)
        err2 = tf.abs(tf.math.real(err2))
        return tf.cast(err1, dtype=tf.float32), tf.cast(err2, dtype=tf.float32)

    def checks(self):
        """ TODO after checking: rewrite with asserts!!!
        Routine for pytest: checking tolerance of manifold functions
        """
        err = self._proj_of_tangent()
        assert err < self.tol, "Projection error for:{}.\
                    ".format(self.descr)

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
        if self.descr[0] not in ['ChoiMatrix', 'DensityMatrix']:
            assert err1 < self.tol, "Rgrad (not in a TMx) error for:{}.\
                    ".format(self.descr)
        if self.descr[1] not in ['log_cholesky']:
            assert err2 < self.tol, "Rgrad (<v1 egrad> != inner<v1 rgrad>) error \
                    for:{}.".format(self.descr)

testdata = [
    ('ChoiMatrix', 'euclidean', manifolds.ChoiMatrix(metric='euclidean'), (4, 4), 1.e-6),
    ('DensityMatrix', 'euclidean', manifolds.DensityMatrix(metric='euclidean'), (4, 4), 1.e-6),
    ('HermitianMatrix', 'euclidean', manifolds.HermitianMatrix(metric='euclidean'), (4, 4), 1.e-6),
    ('PositiveCone', 'log_euclidean', manifolds.PositiveCone(metric='log_euclidean'), (4, 4), 1.e-5),
    ('PositiveCone', 'log_cholesky', manifolds.PositiveCone(metric='log_cholesky'), (4, 4), 1.e-5),
]

@pytest.mark.parametrize("name,metric,manifold,shape,tol", testdata)
def test_manifolds(name, metric, manifold, shape, tol):
    Test = CheckManifolds(manifold, (name, metric), shape, tol)
    Test.checks()
