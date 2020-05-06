Extension of TensorFlow optimizers on Riemannian manifolds that often arise in quantum mechanics (Complex Stiefel manifold, positive-definite cone).
## Basic Example

Here we create an example of the Stiefel manifold with canonical metric and Cayley retraction.
```Python
import QGOpt as qgo
import tensorflow as tf

stiefel_manifold = qgo.manifolds.StiefelManifold(retraction='cayley', metric='canonical')
```
You can create a Riemannian optimizer using the Stiefel manifold above. This optimizer works almost like TF optimizer.
```Python
learning_rate = 0.1
opt = qgo.optimizers.RAdam(stiefel_manifold, learning_rate)  # Riemannian Adam
```
One can create tf.Variable describing point on the Stiefel manifold. tf.Variable must have float dtype, and shape (..., n, m, 2), where (...) enumerates a manifold, n>=m for the Stiefel manifold, last index enumerates the real part [0] and the imag part [1]. You can also use ```real_to_complex``` and ```complex_to_real``` functions to switch between the float representation of a point from a manifold and the complex representation of a point from a manifold.
```Python
U = tf.random.normal((5, 100, 30, 2))  # random initial matrices
U = qgo.manifolds.real_to_complex(U)  # turns matrices to the complex repr. (shape (5, 100, 30, 2) -> (5, 100, 30))
U, _ = tf.linalg.qr(U)  # makes matrices isometric (Stiefel manifold)
U = qgo.manifolds.complex_to_real(U)  # turns matrices back to the float repr. (shape (5, 100, 30) -> (5, 100, 30, 2))
U = tf.Variable(U)  # tf.Variable to be optimized
```
Now you can perform an optimization step of your target function.
```Python
with tf.GradientTape() as tape:
    U_complex = qgo.manifolds.real_to_complex(U)  # turns U to the complex representation
    target = target_function(U_complex)
grad = tape.gradient(target, [U])  # gradient
opt.apply_gradients(zip([grad], [U]))  # optimization step
```
For more examples, see ipython notebooks.

## Installation
```pip install QGOpt```
