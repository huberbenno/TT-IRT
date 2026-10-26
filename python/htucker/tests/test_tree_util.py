import numpy as np
import pytest

from tree.tree import Tree
from tree.tree_util import random_nary_tupletree, zero_root_tupletree, linear_tupletree, \
  balanced_binary_tupletree, weighted_binary_tupletree, random_tupletree


def leaves(tupletree):
  if isinstance(tupletree, tuple):
    return [l for sub in tupletree for l in leaves(sub)]
  return [tupletree]

TREE_KWARGS = {
  'linear': lambda n: {},
  'balanced': lambda n: {},
  'weighted': lambda n: dict(weights=(np.arange(n)+1.)**-1.5),
  'huffmann': lambda n: dict(weights=(np.arange(n)+1.)**-1.5, mode='huffmann'),
  'random': lambda n: dict(seed=2),
}

def zero_root(n, kind):
  tree_type = 'weighted' if kind == 'huffmann' else kind
  return zero_root_tupletree(n, tree_type, **TREE_KWARGS[kind](n))


def test_zero_root_tupletree_values():
  assert zero_root_tupletree(3, 'linear') == (0, (1, (2, (3,))))
  assert zero_root_tupletree(4, 'balanced') == (0, ((1, 2), (3, 4)))
  assert zero_root_tupletree(1, 'balanced') == (0, 1)

@pytest.mark.parametrize('n', [1, 2, 5, 12])
@pytest.mark.parametrize('kind', TREE_KWARGS.keys())
def test_zero_root_tupletree_valid(n, kind):
  tupletree = zero_root(n, kind)
  assert tupletree[0] == 0
  tree = Tree.from_tupletree(tupletree)
  assert tree.order == n + 1

@pytest.mark.parametrize('kind', TREE_KWARGS.keys())
def test_zero_root_tupletree_deterministic(kind):
  assert zero_root(10, kind) == zero_root(10, kind)

def test_zero_root_tupletree_invalid():
  with pytest.raises(ValueError):
    zero_root_tupletree(3, 'foo')
  with pytest.raises(TypeError):
    zero_root_tupletree(3, 'linear', seed=1)
  with pytest.raises(TypeError):
    zero_root_tupletree(3, 'weighted')

@pytest.mark.parametrize('dims', [[7], [5, 2, 9], [3, 0, 8, 1, 4, 6]])
def test_builders_general_dims(dims):
  w = np.linspace(1., .1, len(dims))
  for tupletree in [
    linear_tupletree(dims),
    balanced_binary_tupletree(dims),
    weighted_binary_tupletree(dims, w),
    weighted_binary_tupletree(dims, w, mode='huffmann'),
    random_tupletree(dims, seed=0),
  ]:
    assert sorted(leaves(tupletree)) == sorted(dims)

def test_builders_keep_order():
  dims = [5, 2, 9, 1]
  assert leaves(linear_tupletree(dims)) == dims
  assert leaves(balanced_binary_tupletree(dims)) == dims
  assert leaves(weighted_binary_tupletree(dims, np.ones(4))) == dims

def test_random_tupletree_does_not_mutate():
  dims = [4, 3, 2, 1]
  random_tupletree(dims, seed=0)
  assert dims == [4, 3, 2, 1]


@pytest.mark.parametrize('n', [1, 2, 5, 20])
@pytest.mark.parametrize('max_arity', [2, 3, 5])
@pytest.mark.parametrize('seed', range(5))
def test_random_nary_tupletree_valid(n, max_arity, seed):
  tree = Tree.from_tupletree(random_nary_tupletree(n, max_arity=max_arity, seed=seed))
  assert tree.order == n + 1
  assert sorted(node.dim for node in tree.node_list if node.isleaf) == list(range(n + 1))
  assert all(2 <= node.n_children <= max_arity for node in tree.node_list if not node.isleaf)

def test_random_nary_tupletree_seed():
  assert random_nary_tupletree(10, seed=3) == random_nary_tupletree(10, seed=3)

def test_random_nary_tupletree_general():
  """ Over several seeds, dim 0 is not always a child of the root and arities > 2 occur. """
  trees = [Tree.from_tupletree(random_nary_tupletree(10, max_arity=4, seed=seed)) for seed in range(20)]
  assert any(tree.get_leaf_depths()[0] > 1 for tree in trees)
  assert any(tree.arity > 2 for tree in trees)
