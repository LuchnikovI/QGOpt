"""Riemannian manifolds and additional functions for
converting complex tensors to real with additional
index (which marks real or imag part) and back."""
from QGOpt.manifolds.convert import real_to_complex
from QGOpt.manifolds.convert import complex_to_real
from QGOpt.manifolds.base_manifold import Manifold
from QGOpt.manifolds.stiefel import StiefelManifold
from QGOpt.manifolds.positivecone import PositiveCone

_directly_imported = ['base_manifold', 'convert', 'densm(old)', 'positivecone', 'stiefel']

__all__ = [s for s in dir() if
           s not in _directly_imported and not s.startswith('_')]
