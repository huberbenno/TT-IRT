import numpy as np
import pytest
import torch

from tree.tree import Tree
from tree.tree_util import zero_root_tupletree, random_nary_tupletree
from tree_tensor.tree_tensor_torch import TreeBasedTensor


def make(device='cpu', dtype=torch.float64):
  tree = Tree.from_tupletree(zero_root_tupletree(4, 'balanced'))
  return TreeBasedTensor.randn(tree, (3,) * tree.order, 2, seed=0, dtype=dtype, device=device)

# meta tensors carry a device without data, so device handling is tested without a GPU
@pytest.mark.parametrize('device', ['cpu', 'meta'])
@pytest.mark.parametrize('copy', [TreeBasedTensor, TreeBasedTensor.copy])
def test_copy_keeps_device(device, copy):
  tensor = make(device)
  res = copy(tensor)
  assert res.device == torch.device(device)
  assert all(core.device == torch.device(device) for core in res.cores)

def test_copy_to_device():
  res = TreeBasedTensor(make('cpu'), device='meta')
  assert res.device == torch.device('meta')
  assert all(core.device == torch.device('meta') for core in res.cores)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('copy', [TreeBasedTensor, TreeBasedTensor.copy])
def test_copy_keeps_dtype(dtype, copy):
  res = copy(make(dtype=dtype))
  assert res.dtype == dtype
  assert all(core.dtype == dtype for core in res.cores)

def test_copy_to_dtype():
  res = TreeBasedTensor(make(dtype=torch.float32), dtype=torch.float64)
  assert res.dtype == torch.float64
  assert all(core.dtype == torch.float64 for core in res.cores)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_from_torch_cores_keeps_dtype(dtype):
  tensor = make(dtype=dtype)
  res = TreeBasedTensor(list(tensor.cores), tensor.tree)
  assert res.dtype == dtype and res.device == torch.device('cpu')
  assert res.shape == tensor.shape

@pytest.mark.parametrize('np_dtype, dtype', [(np.float32, torch.float32), (np.float64, torch.float64)])
def test_from_numpy_cores(np_dtype, dtype):
  tensor = make()
  cores = [core.numpy().astype(np_dtype) for core in tensor.cores]
  res = TreeBasedTensor(cores, tensor.tree)
  assert res.dtype == dtype and res.device == torch.device('cpu')
  assert res.shape == tensor.shape
  assert all(torch.allclose(x, y.to(dtype)) for x, y in zip(res.cores, tensor.cores))
  # the cores are copied, not shared with the numpy arrays
  cores[0][:] = 0
  assert not torch.all(res.cores[0] == 0)

def test_unsupported_input():
  with pytest.raises(NotImplementedError):
    TreeBasedTensor(torch.zeros(3), make().tree)


# ---------- arithmetic ----------

TREES = {
  'balanced': zero_root_tupletree(4, 'balanced'),
  'linear': zero_root_tupletree(4, 'linear'),
  'random': zero_root_tupletree(5, 'random', seed=1),
  'nary': random_nary_tupletree(6, max_arity=4, seed=5), # arity 4, dim 0 at depth 3
}
# distinct mode sizes, so that mixed up axes show up
SIZES = (4, 2, 3, 5, 2, 3, 4)
RX, RY = 2, 3

def make_random(tree, rank, seed, dtype=torch.float64, device=None, shape=None):
  if shape is None: shape = SIZES[:tree.order]
  return TreeBasedTensor.randn(tree, shape, rank, seed=seed, dtype=dtype, device=device)

def operands(tree_type, **kwargs):
  """ Two tensors on the same tree with different ranks. """
  tree = Tree.from_tupletree(TREES[tree_type])
  return make_random(tree, RX, seed=0, **kwargs), make_random(tree, RY, seed=1, **kwargs)

def ranks(tensor):
  """ Rank of every edge (last axis of every core except the root's). """
  return [tensor.cores[node].shape[-1] for node in tensor.tree.node_list if not node.isroot]

def assert_dense(tensor, expected, rtol=1e-12, atol=1e-12):
  res = tensor.to_torch()
  assert res.shape == expected.shape
  assert torch.allclose(res, expected, rtol=rtol, atol=atol)

def snapshot(tensor):
  return [core.clone() for core in tensor.cores]

def unchanged(tensor, cores):
  return all(torch.equal(x, y) for x, y in zip(tensor.cores, cores))


@pytest.mark.parametrize('tree_type', TREES.keys())
def test_scalar_mul(tree_type):
  x, _ = operands(tree_type)
  X = x.to_torch()
  for res in [x * 2.5, 2.5 * x, x * torch.tensor(2.5, dtype=torch.float64)]:
    assert_dense(res, 2.5 * X)
    assert ranks(res) == ranks(x)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_scalar_imul(tree_type):
  x, _ = operands(tree_type)
  X = x.to_torch()
  z = x.copy()
  z_id = id(z)
  z *= 2.5
  assert id(z) == z_id
  assert_dense(z, 2.5 * X)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_hadamard(tree_type):
  x, y = operands(tree_type)
  X, Y = x.to_torch(), y.to_torch()
  for res in [x * y, y * x]:
    assert_dense(res, X * Y)
    assert ranks(res) == [rx * ry for rx, ry in zip(ranks(x), ranks(y))]

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_hadamard_imul(tree_type):
  x, y = operands(tree_type)
  X, Y = x.to_torch(), y.to_torch()
  z = x.copy()
  z_id = id(z)
  z *= y
  assert id(z) == z_id
  assert_dense(z, X * Y)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_add(tree_type):
  x, y = operands(tree_type)
  X, Y = x.to_torch(), y.to_torch()
  res = x + y
  assert_dense(res, X + Y)
  assert ranks(res) == [rx + ry for rx, ry in zip(ranks(x), ranks(y))]
  assert res.cores[res.tree.root].shape[-1] == 1

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_iadd(tree_type):
  x, y = operands(tree_type)
  X, Y = x.to_torch(), y.to_torch()
  z = x.copy()
  z_id = id(z)
  z += y
  assert id(z) == z_id
  assert_dense(z, X + Y)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_sub_neg(tree_type):
  x, y = operands(tree_type)
  X, Y = x.to_torch(), y.to_torch()
  assert_dense(x - y, X - Y)
  assert_dense(-x, -X)
  assert_dense(x - x, torch.zeros_like(X), atol=1e-12 * torch.linalg.norm(X).item())
  z = x.copy()
  z_id = id(z)
  z -= y
  assert id(z) == z_id
  assert_dense(z, X - Y)

@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('op', ['scalar_mul', 'neg', 'mul', 'add', 'sub'])
def test_operands_unchanged(tree_type, op):
  x, y = operands(tree_type)
  x_cores, y_cores = snapshot(x), snapshot(y)
  {'scalar_mul': lambda: x * 2., 'neg': lambda: -x, 'mul': lambda: x * y,
   'add': lambda: x + y, 'sub': lambda: x - y}[op]()
  assert unchanged(x, x_cores)
  assert unchanged(y, y_cores)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_dot_norm(tree_type):
  x, y = operands(tree_type)
  X, Y = x.to_torch(), y.to_torch()
  assert float(x.dot(y)) == pytest.approx(torch.sum(X * Y).item(), rel=1e-12)
  assert float(y.dot(x)) == pytest.approx(float(x.dot(y)), rel=1e-12)
  assert float(x.norm()) == pytest.approx(torch.linalg.norm(X).item(), rel=1e-12)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_chained(tree_type):
  x, y = operands(tree_type)
  X, Y = x.to_torch(), y.to_torch()
  assert_dense((x + y) * x, (X + Y) * X)

def test_invalid_operand_type():
  x, _ = operands('balanced')
  with pytest.raises(TypeError):
    x + 1.0
  with pytest.raises(TypeError):
    x * 'a'

@pytest.mark.parametrize('op', ['add', 'mul', 'dot'])
def test_mismatch(op):
  fun = {'add': lambda a, b: a + b, 'mul': lambda a, b: a * b, 'dot': lambda a, b: a.dot(b)}[op]
  x, _ = operands('balanced')
  # same order, different tree
  other_tree = make_random(Tree.from_tupletree(TREES['linear']), RY, seed=1)
  with pytest.raises(AssertionError):
    fun(x, other_tree)
  # same tree, different shape
  other_shape = make_random(x.tree, RY, seed=1, shape=tuple(s + 1 for s in x.shape))
  with pytest.raises(AssertionError):
    fun(x, other_shape)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_keeps_dtype(tree_type):
  x, y = operands(tree_type, dtype=torch.float32)
  X, Y = x.to_torch(), y.to_torch()
  for res, expected in [(x * y, X * Y), (x + y, X + Y), (x - y, X - Y), (-x, -X), (2.5 * x, 2.5 * X)]:
    assert res.dtype == torch.float32
    assert all(core.dtype == torch.float32 for core in res.cores)
    assert_dense(res, expected, rtol=1e-5, atol=1e-5)

# meta tensors carry a device without data, so device handling is tested without a GPU
@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('op', ['mul', 'add'])
def test_keeps_device(tree_type, op):
  x, y = operands(tree_type, device='meta')
  res = x * y if op == 'mul' else x + y
  assert res.device == torch.device('meta')
  assert all(core.device == torch.device('meta') for core in res.cores)
