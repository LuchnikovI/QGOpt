import tensorflow as tf
import QGOpt as qgo
import math
import pytest

#---------------------------------------------------------------------------------#
ham_dim = 20  # dimension of a hamiltonian
renorm_ham_dim = 10  # dimension of a renormalized hamiltonian
number_of_steps = tf.constant(200, dtype=tf.int32)  # number of optimization steps
#---------------------------------------------------------------------------------#

# hamiltonian generation
herm = qgo.manifolds.HermitianMatrix()
h = herm.random((ham_dim, ham_dim), dtype=tf.complex128)

# complex Stiefel manifold
stiefel = qgo.manifolds.StiefelManifold()

# initial random isometric matrix
q = stiefel.random((ham_dim, renorm_ham_dim), dtype=tf.complex128)
q = qgo.manifolds.complex_to_real(q)
q = tf.Variable(q)

# optimizers
opts = {'GD':qgo.optimizers.RSGD(stiefel, 0.05),
        'momentum_GD':qgo.optimizers.RSGD(stiefel, 0.1, 0.9),
        'Nesterov_momentum_GD':qgo.optimizers.RSGD(stiefel, 0.1, 0.9, use_nesterov=True),
        'Adam':qgo.optimizers.RAdam(stiefel, 0.2),
        'AmsGrad':qgo.optimizers.RAdam(stiefel, 0.2, ams=True)}

# exact solution of the problem
exact_solution = tf.math.real(tf.reduce_sum(tf.linalg.eigvalsh(h)[:renorm_ham_dim]))

# optimization function
def optimize(q, h, number_of_steps):
    i = tf.constant(0, dtype=tf.int32)
    loss = tf.constant(0, dtype=tf.float64)
    def body(i, loss):
        with tf.GradientTape() as tape:
            qc = qgo.manifolds.real_to_complex(q)
            loss = tf.math.real(tf.linalg.trace(tf.linalg.adjoint(qc) @ h @ qc))
        grad = tape.gradient(loss, q)
        opt.apply_gradients(zip([grad], [q]))
        return i + 1, loss
    cond = lambda i, loss: i < number_of_steps
    _, loss = tf.while_loop(cond, body, [i, loss])
    return loss

# optimization loops
err_dict = {}
for key, opt in opts.items():
    loss = optimize(q, h, number_of_steps)
    assert loss < 1.e-6, "Optimizer fails:".format(opt)
