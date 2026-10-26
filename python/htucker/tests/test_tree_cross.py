import numpy as np
import pytest

from tree.tree import Tree
from tree.tree_util import linear_tupletree, balanced_binary_tupletree, weighted_binary_tupletree, \
  random_tupletree, random_nary_tupletree


# trees over shuffled dims, so dim 0 is not a child of the root and the leaf order
# differs from the dimension order
TREES = {
  'balanced': balanced_binary_tupletree([2, 4, 3, 0, 1]),
  'linear': linear_tupletree([4, 3, 2, 1, 0]),
  'weighted': weighted_binary_tupletree([3, 2, 5, 4, 0, 1], 0.5**np.arange(6), mode='huffmann'),
  'random': random_tupletree(range(6), seed=1),
  'nary': random_nary_tupletree(6, max_arity=4, seed=5), # arity 4, converges slower
}
# distinct mode sizes, so that mixed up axes show up
SIZES = (4, 5, 3, 6, 4, 5, 3)
RANK = 2
# enough sweeps for the arity 4 tree, binary trees converge in one or two
N_ITER = 15

def make_tree(tree_type):
  tree = Tree.from_tupletree(TREES[tree_type])
  return tree, SIZES[:tree.order]

def rel_err(x, X):
  return np.linalg.norm(x.numpy() - X) / np.linalg.norm(X)

class Recorder:
  """
  eval_f for TreeCross from a numpy function on index arrays. Records the index
  arrays of all calls.
  """
  def __init__(self, fun, backend):
    self.fun = fun
    self.backend = backend
    self.calls = []

  def __call__(self, ind):
    self.calls.append(ind.copy())
    return self.backend.from_numpy(self.fun(ind))

def dense_target(backend, tree, shape):
  """ Dense array of a random rank RANK tensor and an eval_f for its entries. """
  X = backend.randn(tree, shape, RANK, seed=0).numpy()
  return X, Recorder(lambda ind: X[tuple(ind.T)], backend)

def smooth_target(backend, shape):
  """ Dense array of 1 / (1 + sum_d x_d) on a uniform grid and an eval_f for its entries. """
  grid = [np.linspace(0., 1., n) for n in shape]
  X = 1. / (1. + sum(np.meshgrid(*grid, indexing='ij')))
  fun = lambda ind: 1. / (1. + sum(g[ind[:, d]] for d, g in enumerate(grid)))
  return X, Recorder(fun, backend)

def setup_cross(backend, tree_type, target=dense_target, dtype=None):
  tree, shape = make_tree(tree_type)
  X, eval_f = target(backend, tree, shape) if target is dense_target else target(backend, shape)
  kwargs = {} if dtype is None else dict(dtype=dtype)
  x = backend.randn(tree, shape, RANK, seed=1, **kwargs)
  return backend.TC(x), x, X, eval_f

def n_sweeps(capsys):
  """ Number of sweeps reported with verbose=1 since the last call. """
  return capsys.readouterr().out.count('swp:')


@pytest.mark.parametrize('tree_type', TREES.keys())
def test_exact_recovery(backend, tree_type):
  cross, x, X, eval_f = setup_cross(backend, tree_type)
  cross.run(eval_f, n_iter=N_ITER)
  assert rel_err(x, X) < 1e-10

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_eval_f_indices(backend, tree_type):
  cross, x, X, eval_f = setup_cross(backend, tree_type)
  cross.run(eval_f, n_iter=2)
  assert len(eval_f.calls) > 0
  for ind in eval_f.calls:
    assert isinstance(ind, np.ndarray)
    assert np.issubdtype(ind.dtype, np.integer)
    assert ind.ndim == 2 and ind.shape[1] == x.ndim
    assert np.all(ind >= 0) and np.all(ind < np.array(x.shape))
  assert cross.n_eval == sum(ind.shape[0] for ind in eval_f.calls)

@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('eps', [1e-3, 1e-6, 1e-9])
def test_smooth_function(backend, tree_type, eps):
  cross, x, X, eval_f = setup_cross(backend, tree_type, target=smooth_target)
  cross.run(eval_f, n_iter=20, eps=eps)
  assert rel_err(x, X) < eps

@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('eps', [1e-3, 1e-6, 1e-9])
def test_smooth_function_resample(backend, tree_type, eps):
  cross, x, X, eval_f = setup_cross(backend, tree_type, target=smooth_target)
  cross.run(eval_f, n_iter=20, eps=eps, resample=True)
  assert rel_err(x, X) < eps

def test_early_stopping(backend, capsys):
  cross, x, X, eval_f = setup_cross(backend, 'balanced')
  eps = 1e-8
  cross.run(eval_f, n_iter=20, eps=eps, verbose=1)
  assert n_sweeps(capsys) < 20
  assert cross.max_dx < eps

def test_no_early_stopping_eps0(backend, capsys):
  cross, x, X, eval_f = setup_cross(backend, 'balanced')
  cross.run(eval_f, n_iter=3, eps=0., verbose=1)
  assert n_sweeps(capsys) == 3

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_resample(backend, tree_type):
  cross, x, X, eval_f = setup_cross(backend, tree_type)
  cross.run(eval_f, n_iter=N_ITER, resample=True)
  assert rel_err(x, X) < 1e-10

  # for a binary root, resample samples the interior nodes in addition. A wider root
  # is sampled either way, with resample before the sweep at the smaller ranks.
  if Tree.from_tupletree(TREES[tree_type]).root.n_children == 2:
    cross_ref, _, _, eval_f_ref = setup_cross(backend, tree_type)
    cross_ref.run(eval_f_ref, n_iter=1)
    cross_res, _, _, eval_f_res = setup_cross(backend, tree_type)
    cross_res.run(eval_f_res, n_iter=1, resample=True)
    assert cross_res.n_eval > cross_ref.n_eval

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_root_sampling(backend, tree_type, capsys):
  """ Without resample, the root core is sampled only if it has more than two children. """
  cross, x, X, eval_f = setup_cross(backend, tree_type)
  cross.run(eval_f, n_iter=2, verbose=2)
  root_sampled = f'node {x.tree.root.id}, res' in capsys.readouterr().out
  assert root_sampled == (x.tree.root.n_children > 2)

def test_force_init(backend, monkeypatch):
  cross, x, X, eval_f = setup_cross(backend, 'nary')
  n_init = 0
  init = cross._init
  def counting_init():
    nonlocal n_init
    n_init += 1
    init()
  monkeypatch.setattr(cross, '_init', counting_init)

  cross.run(eval_f, n_iter=1)
  assert n_init == 1
  err_1 = rel_err(x, X)
  # continue from the current index sets
  cross.run(eval_f, n_iter=N_ITER, force_init=False)
  assert n_init == 1
  assert rel_err(x, X) < min(1e-10, err_1)
  # default reinitializes
  cross.run(eval_f, n_iter=1)
  assert n_init == 2

def test_n_eval_accumulates(backend):
  cross, x, X, eval_f = setup_cross(backend, 'balanced')
  cross.run(eval_f, n_iter=1)
  n_first = cross.n_eval
  assert n_first > 0
  cross.run(eval_f, n_iter=1)
  assert cross.n_eval > n_first
  assert cross.n_eval == sum(ind.shape[0] for ind in eval_f.calls)

def test_updates_tensor_in_place(backend):
  cross, x, X, eval_f = setup_cross(backend, 'balanced')
  assert cross.tensor is x
  assert rel_err(x, X) > 1e-1
  cross.run(eval_f, n_iter=2)
  assert cross.tensor is x
  assert rel_err(x, X) < 1e-10

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_keeps_dtype(backend, tree_type):
  # eval_f returns float64, the tensor stays float32
  cross, x, X, eval_f = setup_cross(backend, tree_type, target=smooth_target, dtype=backend.float32)
  cross.run(eval_f, n_iter=10, eps=1e-4)
  assert x.dtype == backend.float32
  assert all(core.dtype == backend.float32 for core in x.cores)
  assert rel_err(x, X) < 1e-4
