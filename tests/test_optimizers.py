import pytest
from QGOpt import manifolds
from tests.test_manifolds import CheckManifolds

@pytest.fixture(params=['StiefelManifold'])
def stiefel_name(request):
    return request.param

@pytest.fixture(params=[(8, 4)])
def stiefel_shape(request):
    return request.param

@pytest.fixture(params=[1.e-6])
def stiefel_tol(request):
    return request.param

@pytest.fixture(params=['euclidean', 'canonical'])
def stiefel_metric(request):
    return request.param

@pytest.fixture(params=['svd', 'cayley', 'qr'])
def stiefel_retraction(request):
    return request.param

def test_stiefel_manifold(stiefel_name, stiefel_metric, stiefel_retraction, stiefel_shape, stiefel_tol):
    Test = CheckManifolds(
        manifolds.StiefelManifold(metric=stiefel_metric, retraction=stiefel_retraction),
        (stiefel_name, stiefel_metric),
        stiefel_shape,
        stiefel_tol
    )
    Test.checks()
