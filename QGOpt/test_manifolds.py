import QGOpt.manifolds as m
import pytest
import tensorflow as tf

# just to insure Pytest works
def test_convert(nx=2,ny=2, tol=1.e-8):
    cm = tf.random.uniform([nx, ny, 2])
    assert tf.reduce_sum(cm - m.complex_to_real(m.real_to_complex(cm))) < tol
'''
def test_stiefel():
    m_svd = m.StiefelManifold()
    u = tf.complex(tf.random.normal((n, k), dtype=tf.float64),
                      tf.random.normal((n, k), dtype=tf.float64))
    u, _ = tf.linalg.qr(u)
    v = tf.complex(tf.random.normal((n, k), dtype=tf.float64),
                      tf.random.normal((n, k), dtype=tf.float64))
    zero = tf.complex(tf.zeros((n, k), dtype=tf.float64),
                      tf.zeros((n, k), dtype=tf.float64))

    proj = m_svd.proj(u, v)
    projnorm = m_svd.proj(u, v - proj)
    print(tf.reduce_sum(tf.cast(projnorm, dtype=tf.float64)))

    u_next = m_svd.retraction(u, zero)
    print(tf.reduce_sum(u-u_next))
'''