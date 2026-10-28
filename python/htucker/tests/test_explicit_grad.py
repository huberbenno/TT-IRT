import pytest
import torch

from tree.tree import Tree
from tree.tree_util import zero_root_tupletree, random_nary_tupletree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree_tensor_training.training import ARGS_tree_tensor_training, ExplicitGradTreeTensor, optimize_core, lstsq_core, \
  orthonormalize_phys_leaf, reduced_sweep


TREES = {
  'balanced': zero_root_tupletree(7, 'balanced'),
  'linear': zero_root_tupletree(5, 'linear'),
  'random': zero_root_tupletree(7, 'random', seed=1),
  'nary': random_nary_tupletree(6, max_arity=4, seed=5), # arity 4, dim 0 at depth 3
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
def test_forward_matches_entries(tree_type):
  """ The forward pass gives the fibres along the physical dimension at the sample indices. """
  tree, tensor, ind, _ = setup(TREES[tree_type])
  with torch.no_grad():
    f_ref = torch.stack([torch.ravel(tensor[:, *[int(i) for i in y[1:]]]) for y in ind])
    f = ExplicitGradTreeTensor(tensor).forward(ind)
  assert torch.allclose(f, f_ref, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_grad_matches_autograd(tree_type):
  """ Explicit gradients of the MSE are those of autograd through the (differentiable) forward pass. """
  tree, tensor, ind, target = setup(TREES[tree_type])
  model = ExplicitGradTreeTensor(tensor)
  for core in model.tensor.cores: core.requires_grad_()
  loss = torch.nn.functional.mse_loss(model.forward(ind), target)
  loss.backward()

  with torch.no_grad():
    f = model.forward(ind)
    R = 2 * (f - target) / f.numel()

    for node, E in model.dfs():
      G = model.local_grad(node, R, E)
      G_ref = model.tensor.cores[node].grad
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
@pytest.mark.parametrize('optimizer', ['adam', 'sd', 'lbfgs', 'lstsq'])
def test_optimize_core_decreases_loss(tree_type, optimizer):
  tree, tensor, ind, target = setup(TREES[tree_type])
  model = ExplicitGradTreeTensor(tensor)
  args = ARGS_tree_tensor_training(optimizer=optimizer, epoch_inner=10)

  def local_loss(node, E):
    f = model.local_apply(node, model.tensor.cores[node], E)
    return torch.mean(torch.square(f - target)).item()

  with torch.no_grad():
    model.forward(ind)
    # all nodes on the path from the root to a parameter leaf (index_add_ gradient path)
    leaf = tree.dim2leaf(tree.order - 1)
    assert leaf.dim != model.phys_dim
    path = [leaf]
    while not path[-1].isroot: path.append(path[-1].parent)
    E = None
    for node in reversed(path):
      E = model.env(node, E)
      loss_init = local_loss(node, E)
      optimize_core(model, node, E, target, args, lr=1e-2)
      assert local_loss(node, E) < loss_init, f'loss not decreased at node {node.id}'

def dense_local_problem(model, node, E, target):
  """ Dense matrix of the local problem (columns are the images of the unit cores) and its least squares solution. """
  core = model.tensor.cores[node]
  A = torch.stack([model.local_apply(node, e.reshape(core.shape), E).ravel()
                   for e in torch.eye(core.numel(), dtype=core.dtype)], dim=1)
  x_ref = torch.linalg.lstsq(A, target.ravel().unsqueeze(1), driver='gelsd').solution.ravel()
  return A, x_ref

@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('precondition', [False, True])
def test_lstsq_core_matches_dense_lstsq(tree_type, precondition):
  """ With as many CGLS iterations as core entries, the core is the least squares solution of the dense problem. """
  tree, tensor, ind, target = setup(TREES[tree_type], N=50, rank=2)
  model = ExplicitGradTreeTensor(tensor)
  with torch.no_grad():
    model.forward(ind)
    for node, E in model.dfs(update_up=True):
      core = model.tensor.cores[node]
      A, x_ref = dense_local_problem(model, node, E, target)
      lstsq_core(model, node, E, target, n_iter=2 * core.numel(), precondition=precondition)
      # the problem may be rank deficient, so compare residuals rather than solutions
      res = torch.linalg.norm(A @ core.ravel() - target.ravel())
      res_ref = torch.linalg.norm(A @ x_ref - target.ravel())
      assert res <= res_ref * (1 + 1e-6), f'residual {res:.6e} above least squares residual {res_ref:.6e} at node {node.id}'

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_orthonormalize_phys_leaf_keeps_tensor(tree_type):
  tree, tensor, ind, _ = setup(TREES[tree_type])
  model = ExplicitGradTreeTensor(tensor)
  with torch.no_grad():
    f = model.forward(ind)
    leaf, Q = orthonormalize_phys_leaf(model)
    assert torch.allclose(Q.T @ Q, torch.eye(Q.shape[1], dtype=Q.dtype), atol=1e-12)
    assert torch.allclose(model.forward(ind), f, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_reduced_problem_has_full_gradients(tree_type):
  """ With an orthonormal physical leaf Q, the problem with leaf identity and target y Q has the gradients of the full one. """
  tree, tensor, ind, target = setup(TREES[tree_type])
  full = ExplicitGradTreeTensor(tensor)
  with torch.no_grad():
    leaf, Q = orthonormalize_phys_leaf(full)
    reduced = ExplicitGradTreeTensor(full.tensor)
    reduced.tensor.cores[leaf] = torch.eye(Q.shape[1], dtype=Q.dtype)
    n = target.numel()
    R_full = 2 * (full.forward(ind) - target) / n
    R_red = 2 * (reduced.forward(ind) - target @ Q) / n
    for (node, E_full), (_, E_red) in zip(full.dfs(), reduced.dfs()):
      if node.id == leaf.id: continue
      G_full = full.local_grad(node, R_full, E_full)
      G_red = reduced.local_grad(node, R_red, E_red)
      assert torch.allclose(G_red, G_full, rtol=1e-10, atol=1e-14), f'gradient mismatch at node {node.id}'

@pytest.mark.parametrize('tree_type', TREES.keys())
@pytest.mark.parametrize('optimizer, precondition', [('adam', False), ('sd', False), ('lbfgs', False), ('lstsq', False), ('lstsq', True)])
def test_reduced_sweep_decreases_loss(tree_type, optimizer, precondition):
  tree, tensor, ind, target = setup(TREES[tree_type])
  model = ExplicitGradTreeTensor(tensor)
  args = ARGS_tree_tensor_training(optimizer=optimizer, epoch_inner=5, lstsq_precondition=precondition)
  with torch.no_grad():
    loss_init = torch.nn.functional.mse_loss(model.forward(ind), target)
    model.set_indices(ind)
    reduced_sweep(model, target, args, lr=1e-2)
    assert torch.nn.functional.mse_loss(model.forward(ind), target) < loss_init

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_preconditioned_lstsq_solves_leaves_in_one_iteration(tree_type):
  """ Parameter leaves are solved exactly, for the physical leaf the Kronecker preconditioner is exact. """
  tree, tensor, ind, target = setup(TREES[tree_type], N=50, rank=2)
  model = ExplicitGradTreeTensor(tensor)
  with torch.no_grad():
    model.forward(ind)
    for node, E in model.dfs(update_up=True):
      if not node.isleaf: continue
      core = model.tensor.cores[node]
      A, x_ref = dense_local_problem(model, node, E, target)
      lstsq_core(model, node, E, target, n_iter=1, precondition=True)
      res = torch.linalg.norm(A @ core.ravel() - target.ravel())
      res_ref = torch.linalg.norm(A @ x_ref - target.ravel())
      assert res <= res_ref * (1 + 1e-8), f'residual {res:.6e} above least squares residual {res_ref:.6e} at leaf {node.id}'
