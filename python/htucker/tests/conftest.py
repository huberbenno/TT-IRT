import pytest

from backends import NumpyBackend, TorchBackend


@pytest.fixture(params=[NumpyBackend, TorchBackend], ids=['numpy', 'torch'])
def backend(request):
  return request.param
