QGOpt's documentation
=====================

QGOpt is an extension of TensorFlow optimizers on Riemannian manifolds that often arise in quantum mechanics. QGOpt allows to perform optimization on the following manifolds:

* Complex Stiefel manifold;

* Manifold of density matrices;

* Manifold of Choi matrices;

* Manifold of Hermitian matrices;

* Complex positive-definite cone;

* Manfiold of POVMs.

QGOpt includes Riemannian versions of popular first-order optimization algorithms that are used in deep learning.

One can use this library to perform quantum tomography of states and channels, to solve quantum control problems and optimize quantum unitary circuits, to perform entanglement renormalization, to solve different model identification problems, to optimize tensor networks with natural "quantum" constraints, etc.

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation

   quick_start

   api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   entanglement_renormalization
   channel_tomography
   state_tomography
   optimal_povm
