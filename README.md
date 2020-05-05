Extension of TensorFlow optimizers on Riemannian manifolds that often arise in quantum information processing (Complex Stiefel manifold, positive-definite cone).
# Basic Example

Here we create an example of the Stiefel manifold with canonical metric and Cayley retraction.
```Python
import QGOpt as qgo
import tensorflow as tf

stiefel_manifold = qgo.manifolds.StiefelManifold(retraction='cayley', metric='canonical')
```
