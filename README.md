[![Documentation Status](https://readthedocs.org/projects/qgopt/badge/?version=latest)](https://qgopt.readthedocs.io/en/latest/?badge=latest)

Extension of TensorFlow optimizers on Riemannian manifolds that often arise in quantum mechanics (the Complex Stiefel manifold, the manifold of Choi matrices, etc).

## Documentation

The documentation is available here https://qgopt.readthedocs.io

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
For more examples, see ipython notebooks and documentation.

## Types of manifolds

The current version of the package includes six types of manifolds: the complex Stiefel manifold, the manifold of density matrices, the manifold of Choi matrices, the manifold of Hermitian matrices, the manifold of POVMs and the manifold of positive-definite matrices (positive-definite cone).
```Python
stiefel = qgo.manifolds.StiefelManifold()
density_matrix = qgo.manifolds.DensityMatrix()
choi_matrix = qgo.manifolds.ChoiMatrix()
hermitian_matrix = qgo.manifolds.HermitianMatrix()
positive_cone = qgo.manifolds.PositiveCone()
povm = qgo.manifolds.POVM()
```
For some manifolds, one can also choose a type of reaction and metric.

## Types of optimizers

The current version of the package includes Riemannian versions of popular first-order optimizers that are used in Deep Learning (for more information please read arXiv:1810.00760, arXiv:2002.01113).
```Python
lr = 0.01  # learning rate
m = qgo.manifolds.StiefelManifold()  # example of a manifold
momentum = 0.9

gd_optimizer = qgo.optimizers.RSGD(m, lr)
gd_with_momentum_optimizer = qgo.optimizers.RSGD(m, lr, momentum)
nesterov_gd_with_momentum_optimizer = qgo.optimizers.RSGD(m, lr, momentum, use_nesterov=True)
adam_optimizer = qgo.optimizers.RAdam(m, lr)
amsgrad_optimizer = qgo.optimizers.RAdam(m, lr, ams=True)
```

## Installation
Make sure you have TensorFlow >= 2.0. One can install the package from GitHub (is recommended)

```pip install git+https://github.com/LuchnikovI/QGOpt```

or from pypi (might be different in comparison with the current state of master)

```pip install QGOpt```

## Paper
We have a tutorial [paper](https://arxiv.org/abs/2011.01894). If you use QGOpt we kindly ask you to cite this paper
```
@article{luchnikov2020qgopt,
  title={QGOpt: Riemannian optimization for quantum technologies},
  author={Luchnikov, IA and Ryzhov, A and Filippov, SN and Ouerdane, H},
  journal={arXiv preprint arXiv:2011.01894},
  year={2020}
}
```
