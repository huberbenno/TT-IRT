import numpy as np
import pytest
import torch

from tree.tree import Tree
from tree.tree_util import balanced_binary_tupletree, linear_tupletree, random_tupletree
from tree_tensor.tree_tensor_torch import TreeBasedTensor as TreeBasedTensorTorch
from tree_tensor.tree_tensor import TreeBasedTensor as TreeBasedTensorNumpy


TREES = {
  'balanced': balanced_binary_tupletree(7),
  'linear': linear_tupletree(7),
  'random': random_tupletree(7, seed=1),
}
N_SAMPLES = 2000
BACKENDS = {'torch': TreeBasedTensorTorch, 'numpy': TreeBasedTensorNumpy}

def sample_entries(tensor_cls, tupletree, rank, scale, n_samples=N_SAMPLES, n_entries=3):
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
    T = tensor_cls.randn(tree, shape, rank, seed=s, scale=scale)
    for j, idx in enumerate(entries):
      vals[s, j] = float(T[idx])
  return vals

@pytest.mark.parametrize('backend', BACKENDS.keys())
@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('rank', [2, 10])
@pytest.mark.parametrize('scale', [1., 3.])
def test_randn_entries_centered(backend, tree_type, rank, scale):
  vals = sample_entries(BACKENDS[backend], TREES[tree_type], rank, scale)
  # standard error of the mean is scale / sqrt(N), allow 4 sigma
  tol = 4 * scale / np.sqrt(N_SAMPLES)
  assert np.all(np.abs(vals.mean(axis=0)) < tol), f'means {vals.mean(axis=0)} exceed {tol:.3f}'

@pytest.mark.parametrize('backend', BACKENDS.keys())
@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('scale', [1., 3.])
def test_randn_entries_variance(backend, tree_type, scale):
  # large rank only: for small ranks the entries are heavy-tailed and the sample
  # variance converges too slowly for a reliable test
  vals = sample_entries(BACKENDS[backend], TREES[tree_type], 10, scale)
  rel_var = vals.var(axis=0) / scale**2
  # entries are heavy-tailed (products of normals), so the relative standard error
  # of the sample variance is sqrt((kurtosis - 1) / N) rather than sqrt(2 / N)
  kurt = np.mean(vals**4, axis=0) / np.mean(vals**2, axis=0)**2
  tol = 4 * np.sqrt((kurt - 1) / N_SAMPLES)
  assert np.all(np.abs(rel_var - 1) < tol), f'relative variances {rel_var}, tolerances {tol}'
