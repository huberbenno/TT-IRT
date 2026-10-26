import copy

import numpy as np
import pytest
import torch

from tree.tree import Tree
from tree.tree_util import linear_tupletree, balanced_binary_tupletree, weighted_binary_tupletree, \
  random_tupletree, random_nary_tupletree
from backends import NumpyBackend, TorchBackend


NumpyTT = NumpyBackend.TT
TorchTT = TorchBackend.TT

# dim 0 is not a child of the root and the leaves are not ordered by dimension
BASE_TREE = balanced_binary_tupletree([2, 0, 3, 1, 4])


# ---------- constructor ----------

def test_from_mixed_cores(backend):
  """ Cores of different dtypes raise, unless a dtype is specified to convert them to. """
  tree = Tree.from_tupletree(BASE_TREE)
  cores = list(backend.randn(tree, (3,) * tree.order, 2, seed=0, dtype=backend.float32).cores)
  cores[-1] = backend.from_numpy(np.asarray(cores[-1]).astype(np.float64))
  with pytest.raises(ValueError):
    backend.TT(cores, tree)
  res = backend.TT(cores, tree, dtype=backend.float32)
  assert res.dtype == backend.float32
  assert all(core.dtype == backend.float32 for core in res.cores)

def test_from_nested_lists(backend):
  tree = Tree.from_tupletree(BASE_TREE)
  x = backend.randn(tree, (3,) * tree.order, 2, seed=0)
  lists = [np.asarray(core).tolist() for core in x.cores]
  # float lists are inferred as float64, as by numpy
  res = backend.TT(lists, tree)
  assert res.dtype == backend.float64
  assert res.shape == x.shape
  assert np.array_equal(res.numpy(), x.numpy())
  # converted if specified
  res = backend.TT(lists, tree, dtype=backend.float32)
  assert res.dtype == backend.float32
  assert all(core.dtype == backend.float32 for core in res.cores)

def test_from_mixed_lists_and_arrays(backend):
  tree = Tree.from_tupletree(BASE_TREE)
  x = backend.randn(tree, (3,) * tree.order, 2, seed=0)
  cores = list(x.cores)
  cores[0] = np.asarray(cores[0]).tolist()
  res = backend.TT(cores, tree)
  assert res.dtype == backend.float64
  assert np.array_equal(res.numpy(), x.numpy())

def test_from_int_lists(backend):
  tree = Tree.from_tupletree(BASE_TREE)
  x = backend.TT.ones(tree, (3,) * tree.order)
  int_lists = [np.asarray(core).astype(int).tolist() for core in x.cores]
  res = backend.TT(int_lists, tree)
  assert all(np.asarray(core).dtype == np.int64 for core in res.cores)
  # int and float cores do not share a dtype
  with pytest.raises(ValueError):
    backend.TT(int_lists[:-1] + [list(x.cores)[-1]], tree)

def test_unsupported_input(backend):
  tree = Tree.from_tupletree(BASE_TREE)
  with pytest.raises(NotImplementedError):
    backend.TT(1.0, tree)


# ---------- creation ----------

@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_randn_dtype(backend, dtype_name):
  dtype = getattr(backend, dtype_name)
  tree = Tree.from_tupletree(BASE_TREE)
  x = backend.TT.randn(tree, (3,) * tree.order, 2, seed=0, dtype=dtype)
  assert x.dtype == dtype
  assert all(core.dtype == dtype for core in x.cores)
  assert x.numpy().dtype == np.dtype(dtype_name)

@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_ones_dtype(backend, dtype_name):
  dtype = getattr(backend, dtype_name)
  tree = Tree.from_tupletree(BASE_TREE)
  x = backend.TT.ones(tree, (3,) * tree.order, dtype=dtype)
  assert x.dtype == dtype
  assert all(core.dtype == dtype for core in x.cores)
  X = x.numpy()
  assert X.dtype == np.dtype(dtype_name)
  assert np.all(X == 1)


# ---------- arithmetic ----------

def orderings(n, n_shuffled=3, seed=0):
  """ Distinct orderings of the dims 0..n-1: identity, reversed and n_shuffled seeded shuffles. """
  dims = list(range(n))
  res = [dims, dims[::-1]]
  rng = np.random.default_rng(seed)
  while len(res) < 2 + n_shuffled:
    perm = rng.permutation(n).tolist()
    if perm not in res: res.append(perm)
  return res

# order and builder of each non random tree type
BUILDERS = {
  'balanced': (5, balanced_binary_tupletree),
  'linear': (5, linear_tupletree), # ends in a node with a single child
  'weighted': (6, lambda dims: weighted_binary_tupletree(dims, 0.5**np.arange(len(dims)), mode='huffmann')),
}
# every non random tree type over several orderings of the dims, so that dim 0 is at
# varying depths and the leaf order differs from the dimension order
TREES = {
  f"{name}-{''.join(map(str, dims))}": builder(dims)
  for name, (n, builder) in BUILDERS.items() for dims in orderings(n)
}
TREES['random'] = random_tupletree(range(6), seed=1)
TREES['nary'] = random_nary_tupletree(6, max_arity=4, seed=5) # arity 4, dim 0 at depth 3
# distinct mode sizes, so that mixed up axes show up
SIZES = (4, 2, 3, 5, 2, 3, 4)
RX, RY = 2, 3

def make_random(backend, tree, rank, seed, shape=None, **kwargs):
  if shape is None: shape = SIZES[:tree.order]
  return backend.randn(tree, shape, rank, seed=seed, **kwargs)

def operands(backend, tree_type, **kwargs):
  """ Two tensors on the same tree with different ranks. """
  tree = Tree.from_tupletree(TREES[tree_type])
  return make_random(backend, tree, RX, seed=0, **kwargs), make_random(backend, tree, RY, seed=1, **kwargs)

def ranks(tensor):
  """ Rank of every edge (last axis of every core except the root's). """
  return [tensor.cores[node].shape[-1] for node in tensor.tree.node_list if not node.isroot]

def assert_dense(tensor, expected, rtol=1e-12, atol=1e-12):
  res = tensor.numpy()
  assert res.shape == expected.shape
  assert np.allclose(res, expected, rtol=rtol, atol=atol)

def snapshot(tensor):
  return [copy.deepcopy(core) for core in tensor.cores]

def unchanged(tensor, cores):
  return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(tensor.cores, cores))


@pytest.mark.parametrize('tree_type', TREES.keys())
def test_scalar_mul(backend, tree_type):
  x, _ = operands(backend, tree_type)
  X = x.numpy()
  for res in [x * 2.5, 2.5 * x, x * np.float64(2.5)]:
    assert_dense(res, 2.5 * X)
    assert ranks(res) == ranks(x)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_scalar_imul(backend, tree_type):
  x, _ = operands(backend, tree_type)
  X = x.numpy()
  z = x.copy()
  z_id = id(z)
  z *= 2.5
  assert id(z) == z_id
  assert_dense(z, 2.5 * X)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_hadamard(backend, tree_type):
  x, y = operands(backend, tree_type)
  X, Y = x.numpy(), y.numpy()
  for res in [x * y, y * x]:
    assert_dense(res, X * Y)
    assert ranks(res) == [rx * ry for rx, ry in zip(ranks(x), ranks(y))]

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_hadamard_imul(backend, tree_type):
  x, y = operands(backend, tree_type)
  X, Y = x.numpy(), y.numpy()
  z = x.copy()
  z_id = id(z)
  z *= y
  assert id(z) == z_id
  assert_dense(z, X * Y)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_add(backend, tree_type):
  x, y = operands(backend, tree_type)
  X, Y = x.numpy(), y.numpy()
  res = x + y
  assert_dense(res, X + Y)
  assert ranks(res) == [rx + ry for rx, ry in zip(ranks(x), ranks(y))]
  assert res.cores[res.tree.root].shape[-1] == 1

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_iadd(backend, tree_type):
  x, y = operands(backend, tree_type)
  X, Y = x.numpy(), y.numpy()
  z = x.copy()
  z_id = id(z)
  z += y
  assert id(z) == z_id
  assert_dense(z, X + Y)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_sub_neg(backend, tree_type):
  x, y = operands(backend, tree_type)
  X, Y = x.numpy(), y.numpy()
  assert_dense(x - y, X - Y)
  assert_dense(-x, -X)
  assert_dense(x - x, np.zeros_like(X), atol=1e-12 * np.linalg.norm(X))
  z = x.copy()
  z_id = id(z)
  z -= y
  assert id(z) == z_id
  assert_dense(z, X - Y)

@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('op', ['scalar_mul', 'neg', 'mul', 'add', 'sub'])
def test_operands_unchanged(backend, tree_type, op):
  x, y = operands(backend, tree_type)
  x_cores, y_cores = snapshot(x), snapshot(y)
  {'scalar_mul': lambda: x * 2., 'neg': lambda: -x, 'mul': lambda: x * y,
   'add': lambda: x + y, 'sub': lambda: x - y}[op]()
  assert unchanged(x, x_cores)
  assert unchanged(y, y_cores)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_dot_norm(backend, tree_type):
  x, y = operands(backend, tree_type)
  X, Y = x.numpy(), y.numpy()
  assert x.dot(y).item() == pytest.approx(np.sum(X * Y), rel=1e-12)
  assert y.dot(x).item() == pytest.approx(x.dot(y).item(), rel=1e-12)
  assert x.norm().item() == pytest.approx(np.linalg.norm(X), rel=1e-12)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_chained(backend, tree_type):
  x, y = operands(backend, tree_type)
  X, Y = x.numpy(), y.numpy()
  assert_dense((x + y) * x, (X + Y) * X)

def test_invalid_operand_type(backend):
  x, _ = operands(backend, 'balanced-01234')
  with pytest.raises(TypeError):
    x + 1.0
  with pytest.raises(TypeError):
    x * 'a'

@pytest.mark.parametrize('op', ['add', 'mul', 'dot'])
def test_mismatch(backend, op):
  fun = {'add': lambda a, b: a + b, 'mul': lambda a, b: a * b, 'dot': lambda a, b: a.dot(b)}[op]
  x, _ = operands(backend, 'balanced-01234')
  # same order, different tree
  other_tree = make_random(backend, Tree.from_tupletree(TREES['linear-01234']), RY, seed=1)
  with pytest.raises(AssertionError):
    fun(x, other_tree)
  # same tree, different shape
  other_shape = make_random(backend, x.tree, RY, seed=1, shape=tuple(s + 1 for s in x.shape))
  with pytest.raises(AssertionError):
    fun(x, other_shape)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_keeps_dtype(backend, tree_type):
  x, y = operands(backend, tree_type, dtype=backend.float32)
  X, Y = x.numpy(), y.numpy()
  for res, expected in [(x * y, X * Y), (x + y, X + Y), (x - y, X - Y), (-x, -X), (2.5 * x, 2.5 * X)]:
    assert res.dtype == backend.float32
    assert all(core.dtype == backend.float32 for core in res.cores)
    assert_dense(res, expected, rtol=1e-5, atol=1e-5)


# ---------- random init ----------

# one shuffled ordering per non random tree type: the distribution of the entries
# does not depend on the ordering, and sampling is expensive
RANDN_TREES = [f"{name}-{''.join(map(str, orderings(n)[2]))}" for name, (n, _) in BUILDERS.items()] \
  + ['random', 'nary']
N_SAMPLES = 2000

def sample_entries(backend, tupletree, rank, scale, n_samples=N_SAMPLES, n_entries=3):
  """
  Draw n_samples independent tensors and evaluate each at n_entries fixed indices.
  Returns array of shape (n_samples, n_entries).
  """
  tree = Tree.from_tupletree(tupletree)
  shape = (6,) + tuple(range(2, 2 + tree.order - 1))
  rng = np.random.default_rng(0)
  entries = [tuple(int(i) for i in rng.integers(0, shape)) for _ in range(n_entries)]
  vals = np.zeros((n_samples, n_entries))
  for s in range(n_samples):
    T = backend.TT.randn(tree, shape, rank, seed=s, scale=scale)
    for j, idx in enumerate(entries):
      vals[s, j] = float(T[idx])
  return vals

@pytest.mark.parametrize('tree_type', RANDN_TREES)
@pytest.mark.parametrize('rank', [2, 10])
@pytest.mark.parametrize('scale', [1., 3.])
def test_randn_entries_centered(backend, tree_type, rank, scale):
  vals = sample_entries(backend, TREES[tree_type], rank, scale)
  # standard error of the mean is scale / sqrt(N), allow 4 sigma
  tol = 4 * scale / np.sqrt(N_SAMPLES)
  assert np.all(np.abs(vals.mean(axis=0)) < tol), f'means {vals.mean(axis=0)} exceed {tol:.3f}'

@pytest.mark.parametrize('tree_type', RANDN_TREES)
@pytest.mark.parametrize('scale', [1., 3.])
def test_randn_entries_variance(backend, tree_type, scale):
  # large rank only: for small ranks the entries are heavy-tailed and the sample
  # variance converges too slowly for a reliable test
  vals = sample_entries(backend, TREES[tree_type], 10, scale)
  rel_var = vals.var(axis=0) / scale**2
  # entries are heavy-tailed (products of normals), so the relative standard error
  # of the sample variance is sqrt((kurtosis - 1) / N) rather than sqrt(2 / N)
  kurt = np.mean(vals**4, axis=0) / np.mean(vals**2, axis=0)**2
  tol = 4 * np.sqrt((kurt - 1) / N_SAMPLES)
  assert np.all(np.abs(rel_var - 1) < tol), f'relative variances {rel_var}, tolerances {tol}'


# ---------- torch ----------

def make(device='cpu', dtype=torch.float64):
  tree = Tree.from_tupletree(BASE_TREE)
  return TorchTT.randn(tree, (3,) * tree.order, 2, seed=0, dtype=dtype, device=device)

# meta tensors carry a device without data, so device handling is tested without a GPU
@pytest.mark.parametrize('device', ['cpu', 'meta'])
@pytest.mark.parametrize('copy', [TorchTT, TorchTT.copy])
def test_torch_copy_keeps_device(device, copy):
  tensor = make(device)
  res = copy(tensor)
  assert res.device == torch.device(device)
  assert all(core.device == torch.device(device) for core in res.cores)

def test_torch_copy_to_device():
  res = TorchTT(make('cpu'), device='meta')
  assert res.device == torch.device('meta')
  assert all(core.device == torch.device('meta') for core in res.cores)

def test_torch_from_cores_mixed_devices():
  """ Cores on different devices raise, unless a device is specified to move them to. """
  tensor = make('cpu')
  cores = list(tensor.cores)
  cores[-1] = cores[-1].to('meta')
  with pytest.raises(ValueError):
    TorchTT(cores, tensor.tree)
  res = TorchTT(cores, tensor.tree, device='meta')
  assert res.device == torch.device('meta')
  assert all(core.device == torch.device('meta') for core in res.cores)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('copy', [TorchTT, TorchTT.copy])
def test_torch_copy_keeps_dtype(dtype, copy):
  res = copy(make(dtype=dtype))
  assert res.dtype == dtype
  assert all(core.dtype == dtype for core in res.cores)

def test_torch_copy_to_dtype():
  res = TorchTT(make(dtype=torch.float32), dtype=torch.float64)
  assert res.dtype == torch.float64
  assert all(core.dtype == torch.float64 for core in res.cores)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_torch_from_cores_keeps_dtype(dtype):
  tensor = make(dtype=dtype)
  res = TorchTT(list(tensor.cores), tensor.tree)
  assert res.dtype == dtype and res.device == torch.device('cpu')
  assert res.shape == tensor.shape

@pytest.mark.parametrize('np_dtype, dtype', [(np.float32, torch.float32), (np.float64, torch.float64)])
def test_torch_from_numpy_cores(np_dtype, dtype):
  tensor = make()
  cores = [core.numpy().astype(np_dtype) for core in tensor.cores]
  res = TorchTT(cores, tensor.tree)
  assert res.dtype == dtype and res.device == torch.device('cpu')
  assert res.shape == tensor.shape
  assert all(torch.allclose(x, y.to(dtype)) for x, y in zip(res.cores, tensor.cores))
  # the cores are copied, not shared with the numpy arrays
  cores[0][:] = 0
  assert not torch.all(res.cores[0] == 0)

def test_torch_scalar_mul_tensor_scalar():
  x, _ = operands(TorchBackend, 'balanced-01234')
  assert_dense(x * torch.tensor(2.5, dtype=torch.float64), 2.5 * x.numpy())

# meta tensors carry a device without data, so device handling is tested without a GPU
@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('op', ['mul', 'add'])
def test_torch_keeps_device(tree_type, op):
  x, y = operands(TorchBackend, tree_type, device='meta')
  res = x * y if op == 'mul' else x + y
  assert res.device == torch.device('meta')
  assert all(core.device == torch.device('meta') for core in res.cores)


# ---------- numpy ----------

@pytest.mark.parametrize('copy', [NumpyTT, NumpyTT.copy])
def test_numpy_copy_is_deep(copy):
  tree = Tree.from_tupletree(BASE_TREE)
  x = NumpyTT.randn(tree, (3,) * tree.order, 2, seed=0)
  X = x.numpy()
  res = copy(x)
  for core in res.cores:
    core[...] = 0
  assert np.array_equal(x.numpy(), X)

def test_numpy_from_cores():
  tree = Tree.from_tupletree(BASE_TREE)
  x = NumpyTT.randn(tree, (3,) * tree.order, 2, seed=0)
  cores = [core.copy() for core in x.cores]
  res = NumpyTT(cores, tree)
  assert res.shape == x.shape
  assert np.array_equal(res.numpy(), x.numpy())
  # the cores are copied, not shared with the input arrays
  cores[0][:] = 0
  assert not np.all(res.cores[0] == 0)

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('copy', [NumpyTT, NumpyTT.copy])
def test_numpy_copy_keeps_dtype(dtype, copy):
  tree = Tree.from_tupletree(BASE_TREE)
  res = copy(NumpyTT.randn(tree, (3,) * tree.order, 2, seed=0, dtype=dtype))
  assert res.dtype == dtype
  assert all(core.dtype == dtype for core in res.cores)

def test_numpy_copy_to_dtype():
  tree = Tree.from_tupletree(BASE_TREE)
  x = NumpyTT.randn(tree, (3,) * tree.order, 2, seed=0, dtype=np.float32)
  res = NumpyTT(x, dtype=np.float64)
  assert res.dtype == np.float64
  assert all(core.dtype == np.float64 for core in res.cores)
  assert x.dtype == np.float32

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_numpy_from_cores_dtype(dtype):
  tree = Tree.from_tupletree(BASE_TREE)
  cores = [core.astype(dtype) for core in NumpyTT.randn(tree, (3,) * tree.order, 2, seed=0).cores]
  # inherited from the cores
  assert NumpyTT(cores, tree).dtype == dtype
  # converted if specified
  res = NumpyTT(cores, tree, dtype=np.float32)
  assert res.dtype == np.float32
  assert all(core.dtype == np.float32 for core in res.cores)

def test_numpy_randn_same_values():
  """ The cores are drawn in float64 and converted, so the seed fixes the values. """
  tree = Tree.from_tupletree(BASE_TREE)
  x64 = NumpyTT.randn(tree, (3,) * tree.order, 2, seed=0)
  x32 = NumpyTT.randn(tree, (3,) * tree.order, 2, seed=0, dtype=np.float32)
  assert all(np.array_equal(c32, c64.astype(np.float32)) for c32, c64 in zip(x32.cores, x64.cores))
