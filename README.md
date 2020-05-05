Extension of TensorFlow optimizers on Riemannian manifolds that often arise in quantum information processing (Complex Stiefel manifold, positive-definite cone).
# Basic Example

Here we create an example of the Stiefel manifold with canonical metric and Cayley retraction.
```Python
import QGOpt as qgo
import tensorflow as tf

stiefel_manifold = qgo.manifolds.StiefelManifold(retraction='cayley', metric='canonical')
```
You can create a Riemannian optimizer using the Stiefel manifold above. This optimizer works almost like TF optimizer.
```Python
learning_rate = 0.1
opt = qgo.optimizers.RAdam(stiefel_manifold, learning_rate)
```
