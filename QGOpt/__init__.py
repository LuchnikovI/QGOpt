"""The packege contains classes and methods to extend tf optimizers
on manifolds which friquently appear in the quantum data processing"""
from QGOpt import manifolds
from QGOpt import optimizers

_directly_imported = ['base_manifold', 'convert', 'densm(old)', 'positivecone', 'stiefel', 'Adam', 'SGD']

__all__ = [s for s in dir() if
           s not in _directly_imported and not s.startswith('_')]
