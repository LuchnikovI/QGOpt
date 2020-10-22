Frequently asked questions
==========================

Is there a relation between complex matrix manifolds and real matrix manifolds?
-----------------------------------------------------------------

One can represent any complex matrix :math:`D = E + iF` as a real matrix :math:`\tilde{D} = \begin{pmatrix}
E & F\\
-F & E
\end{pmatrix}`.
Then, matrix operations on matrices without and with tilde
are related as follows:

:math:`A + B \longleftrightarrow \tilde{A} + \tilde{B}, \ AB \longleftrightarrow \tilde{A}\tilde{B}, \ A^\dagger \longleftrightarrow \tilde{A}^T`.

Therefore, any complex manifold has a corresponding real one. For more details read

Sato, H., & Iwai, T. (2013). A Riemannian optimization approach to the matrix singular value decomposition. *SIAM Journal on Optimization*, 23(1), 188-212.

How to perform optimization over complex tensors and matrices?
--------------------------------------------------------------

To perform optimization over complex matrices and tensors, one needs to follow several simple rules. First of all, a value of a loss function, you want to optimize, must be real. Secondly, the class for TensorFlow optimizers works well only with real valued variables. Due to the class for Riemannian optimizers of QGOpt is inherited from the class for TensorFlow optimizers, one requires all input variables to be real. Normally a point from a manifold is represented by a complex matrix or tensor, but one can also consider a point as a real tensor. In general, we suggest the following scheme for variables initialization and optimization:

.. code-block:: python
  
    # Here we initialize an example of the complex Stiefel manifold.
    m = qgo.manifolds.StiefelManifold()
    # Here we initialize a unitary matrix by using an example of the
    # complex Stiefel manifold (dtype = tf.complex64).
    u = m.random((4, 4))
    # Here we turn a complex matrix to its real representation
    # (shape=(4, 4) --> shape=(4, 4, 2)).
    # The last index enumerates real and imaginary parts.
    # (dtype=tf.complex64 --> dtype=tf.float32).
    u = qgo.manifolds.complex_to_real(u)
    # Here we turn u to tf.Variable, any Riemannian optimizer 
    # can perform optimization over u now, because it is
    # real valued TensorFlow variable. Note also, that
    # any Riemannian optimizer preserves all the constraints
    # of a corresponding complex manifold.
    u = tf.Variable(u)

After initialization of variables one can perform optimization step:

.. code-block:: python
    
    lr = 0.01 # optimization step size
    opt = qgo.optimizers.RAdam(m, lr) # optimizer initialization

    # Here we calculate the gradient and perform optimization step.
    # Note, that in the body of a TensorFlow graph one can
    # have complex-valued tensors. It is only important to
    # have input variables and target function to be real.
    tf.with tf.GradientTape() as tape:

       # Here we turn the real representation of a point on a manifold
       # back to the complex representation.
       # (shape=(4, 4, 2) --> shape=(4, 4)),
       # (dtype=tf.float32 --> dtype=tf.complex64)
       uc = qgo.manifolds.real_to_complex(u)

       # Here we calculate the value of a target function, we want to minimize.
       # Target function returns real value. If a target function returns an 
       # imaginary value, then optimizer minimizes real part of a function.
       loss = target_function(uc)

    # Here we calculate the gradient of a function.
    grad = tape.gradient(loss, u)
    # And perform an optimization step.
    opt.apply_gradients(zip([grad], [u]))
