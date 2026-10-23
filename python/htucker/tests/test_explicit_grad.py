import pytest
import torch

from tree.tree import Tree
from tree.tree_util import balanced_binary_tupletree, linear_tupletree, random_tupletree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from benchmarks.tree_tensor_training import HT_Module
from benchmarks.tree_tensor_training_explicit import ExplicitGradTreeTensor, ARGS_tree_tensor_training_explicit, optimize_core


TREES = {
  'balanced': balanced_binary_tupletree(7),
  'linear': linear_tupletree(5),
  'random': random_tupletree(7, seed=1),
}

def setup(tupletree, N=50, rank=3):
  tree = Tree.from_tupletree(tupletree)
  shape = (6,) + tuple(range(2, 2 + tree.order - 1))
  tensor = TreeBasedTensor.randn(tree, shape, rank, seed=0)
  gen = torch.Generator().manual_seed(0)
  ind = torch.stack([torch.randint(0, s, (N,), generator=gen) for s in shape], dim=1)
  target = torch.randn(N, shape[0], dtype=torch.float64, generator=gen)
  return tree, tensor, ind, target

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_forward_matches_autograd_module(tree_type):
  tree, tensor, ind, _ = setup(TREES[tree_type])
  with torch.no_grad():
    f_ref = HT_Module(tensor).forward(ind)
    f = ExplicitGradTreeTensor(tensor).forward(ind)
  assert torch.allclose(f, f_ref, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_grad_matches_autograd(tree_type):
  tree, tensor, ind, target = setup(TREES[tree_type])

  ref = HT_Module(tensor)
  loss = torch.nn.functional.mse_loss(ref.forward(ind), target)
  loss.backward()

  model = ExplicitGradTreeTensor(tensor)
  with torch.no_grad():
    f = model.forward(ind)
    R = 2 * (f - target) / f.numel()

    for node, E in model.dfs():
      G = model.local_grad(node, R, E)
      G_ref = ref.tensor.cores[node].grad
      assert torch.allclose(G, G_ref, rtol=1e-10, atol=1e-14), f'gradient mismatch at node {node.id}'

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_local_apply_is_linear_in_core(tree_type):
  tree, tensor, ind, _ = setup(TREES[tree_type])
  model = ExplicitGradTreeTensor(tensor)
  with torch.no_grad():
    f = model.forward(ind)

    for node, E in model.dfs():
      core = model.tensor.cores[node]
      assert torch.allclose(model.local_apply(node, core, E), f, rtol=1e-12, atol=1e-12)
      D = torch.randn_like(core)
      assert torch.allclose(model.local_apply(node, core + 2 * D, E),
                            f + 2 * model.local_apply(node, D, E), rtol=1e-10, atol=1e-12)

@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('optimizer', ['adam', 'sd', 'lbfgs'])
def test_optimize_core_decreases_loss(tree_type, optimizer):
  tree, tensor, ind, target = setup(TREES[tree_type])
  model = ExplicitGradTreeTensor(tensor)
  args = ARGS_tree_tensor_training_explicit(optimizer=optimizer, epoch_inner=10)

  def local_loss(node, E):
    f = model.local_apply(node, model.tensor.cores[node], E)
    return torch.mean(torch.square(f - target)).item()

  with torch.no_grad():
    model.forward(ind)
    # root and, descending along last children, a parameter leaf (index_add_ gradient path)
    node = tree.root
    E = model.env(node)
    while True:
      loss_init = local_loss(node, E)
      optimize_core(model, node, E, target, args, lr=1e-2)
      assert local_loss(node, E) < loss_init, f'loss not decreased at node {node.id}'
      if node.isleaf:
        assert node.dim != model.phys_dim
        break
      child = node.children[-1]
      E = model.env(child, E)
      node = child
