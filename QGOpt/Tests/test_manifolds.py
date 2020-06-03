import QGOpt.manifolds as manifolds
import pytest
import tensorflow as tf

def test_convert(nx=2,ny=2, tol=1.e-12):
    cm = tf.random.uniform([nx, ny, 2])
    assert tf.reduce_sum(cm - manifolds.complex_to_real(
                            manifolds.real_to_complex(cm))) < tol


def test_stiefel(n=10, k=5, tol=1.e-12):
    '''
    Unit tests for Stiefel manifolds
    - projection
    - retraction
    '''

    '''list of metrics and retractions'''
    list_of_metrics = ['euclidean', 'canonical']
    list_of_retractions = ['svd', 'cayley']

    u = tf.complex(tf.random.normal((n, k), dtype=tf.float64),
                      tf.random.normal((n, k), dtype=tf.float64))
    u, _ = tf.linalg.qr(u)
    v = tf.complex(tf.random.normal((n, k), dtype=tf.float64),
                      tf.random.normal((n, k), dtype=tf.float64))
    zero = tf.complex(tf.zeros((n, k), dtype=tf.float64),
                      tf.zeros((n, k), dtype=tf.float64))

    for metric in list_of_metrics:
        for retraction in list_of_retractions:
            m = manifolds.StiefelManifold(metric=metric, retraction=retraction)

            '''' testing projetcion
            Algorithm:
            1) choose an arbitraty vector and arbitrary orthogonal manifold
            2) calculate projection of the vector to the manifolds
            3) calculate normal vector and project to the manifold
            '''

            proj = m.proj(u, v)
            projnorm = m.proj(u, v - proj)
            out = tf.abs(tf.reduce_sum(tf.cast(projnorm, dtype=tf.float64)))
            assert out < tol, "Projection failed for Stiefel: metric - {}, \
            retraction-{}. Tolerance obtained {:1.1e} > tolerance desired {}\
            ".format(metric, retraction, out, tol)

            '''testing retraction
            ----------------------------------------------------------------
            http://web.math.princeton.edu/~nboumal/book/IntroOptimManifolds_
            Boumal_2020.pdf, chapter 8.7
            ----------------------------------------------------------------
            Let Rx: TxM → M is the restriction of R at x, so that Rx(v) = R(x,v).
            1) Rx(0)=x
            2) DRx(0): TxM→TxM is the identity map: DRx(0)[v] = v.
            (To be clear, here, 0 denotes the zero tangent vector at x,
            that is, the equivalence class of smooth curves on M that pass
            through x at t = 0 with zero velocity.)
            '''

            u_next = m.retraction(u, zero)
            out = tf.reduce_sum(tf.cast(u-u_next, dtype=tf.float64))
            assert out < tol, "retraction failed for Stiefel: metric - {}, \
            retraction-{}. Tolerance obtained {:1.1e} > tolerance desired {}\
            ".format(metric, retraction, out, tol)
            proj_next = m.vector_transport(u, proj, zero)
            out = tf.reduce_sum(tf.cast(proj-proj_next, dtype=tf.float64))
            assert out < tol, "vector_transport failed for Stiefel:\
            metric - {}, retraction-{}. Tolerance obtained {:1.1e} > tolerance \
            desired {}".format(metric, retraction, out, tol)



def test_positivecone(q=10, tol = 1.e-12):
    '''
    Unit tests for positivecone manifolds
    - projection
    - retraction
    '''

    '''list of metrics'''
    list_of_metrics = ['log_cholesky', 'log_euclidean']

    u = tf.random.normal((q, q, 2), dtype=tf.float64)
    u = manifolds.real_to_complex(u)
    u = tf.linalg.adjoint(u) @ u
    u = u / tf.linalg.trace(u)

    v = tf.complex(tf.random.normal((q, q), dtype=tf.float64),
                   tf.random.normal((q, q), dtype=tf.float64))
    zero = tf.complex(tf.zeros((q, q), dtype=tf.float64),
                      tf.zeros((q, q), dtype=tf.float64))
    for metric in list_of_metrics:
        m = manifolds.PositiveCone(metric),
        '''' testing projetcion
        Algorithm:
        1) choose an arbitraty vector and arbitrary orthogonal manifold
        2) calculate projection of the vector to the manifolds
        3) calculate normal vector and project to the manifold
        '''
        proj = m.proj(u, v)
        projnorm = m.proj(u, v - proj)
        out = tf.reduce_sum(tf.cast(projnorm, dtype=tf.float64))
        assert out < tol, "projection failed for PositiveCone: metric - {}.\
        Tol obtained {:1.1e} > tol desired {}".format(metric, out, tol)

        '''testing retraction
        ----------------------------------------------------------------
        http://web.math.princeton.edu/~nboumal/book/IntroOptimManifolds_
        Boumal_2020.pdf, chapter 8.7
        ----------------------------------------------------------------
        Let Rx: TxM → M is the restriction of R at x, so that Rx(v) = R(x,v).
        1) Rx(0)=x
        2) DRx(0): TxM→TxM is the identity map: DRx(0)[v] = v.
        (To be clear, here, 0 denotes the zero tangent vector at x,
        that is, the equivalence class of smooth curves on M that pass
        through x at t = 0 with zero velocity.)
        '''
        u_next = m.retraction(u, zero)
        out = tf.reduce_sum(tf.cast(u-u_next, dtype=tf.float64))
        assert out < tol, "retraction failed for PositiveCone: metric - {}.\
        Tol obtained {:1.1e} > tol desired {}".format(metric, out, tol)
        proj_next = m.vector_transport(u, proj, zero)
        out = tf.reduce_sum(tf.cast(proj-proj_next, dtype=tf.float64))
        assert out < tol, "vector_transport failed for PositiveCone: metric-{}.\
        Tol obtained {:1.1e} > tol desired {}".format(metric, out, tol)
