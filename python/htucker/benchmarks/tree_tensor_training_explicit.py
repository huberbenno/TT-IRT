import numpy as np
from dataclasses import dataclass
import json
from dataclasses import asdict
from benchmarks.util import NumpyEncoder, RMS, coeff_mc_errors

from pde.coeff_function import coeff

import torch
from torch import optim
from tree.tree import Tree, TreeNode, NodeIndexedList
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree.tree_util import zero_root_tupletree


def _contract(*operands, out):
  """
  einsum over (tensor, labels) pairs with arbitrary hashable labels.
  Labels are mapped to local integers, so the number of distinct labels per call
  stays small even for large trees.
  """
  label_map = {}
  args = []
  for t, labels in operands:
    args += [t, [label_map.setdefault(l, len(label_map)) for l in labels]]
  args += [[label_map[l] for l in out]]
  return torch.einsum(*args)

def _free_labels(operands):
  """ Sample ('n') and physical ('p') labels present in any operand. """
  return [l for l in ('n', 'p') if any(l in labels for _, labels in operands)]


class ExplicitGradTreeTensor:
  """
  Evaluates a tree tensor at a batch of parameter indices (the physical dimension
  phys_dim is always evaluated in full) and computes gradients w.r.t. single cores
  explicitly.

  With all other cores fixed, the tensor is linear in the core of node v:
    f = A_v(core_v) = contract(core_v, E[v], U[c] for c in children(v)),
  where U[c] are the subtree contractions of the children and E[v] is the
  environment (everything outside the subtree of v). Hence for the MSE loss
    grad_v = A_v^T (dL/df),
  which is the same contraction with the core replaced by dL/df.
  """
  def __init__(self, tensor: TreeBasedTensor, phys_dim: int = 0) -> None:
    self.tensor = TreeBasedTensor(tensor)
    self.phys_dim = phys_dim
    self.indices = None

  @property
  def tree(self) -> Tree:
    return self.tensor.tree

  def set_indices(self, indices):
    self.indices = torch.atleast_2d(indices)
    self.U = NodeIndexedList(self.tree.n_nodes * [None])

  def _rank_label(self, node: TreeNode):
    return ('r', node.id)

  def _core_op(self, node: TreeNode, core=None):
    """ Core of node as einsum operand (rows selected by sample index for parameter leafs). """
    if core is None: core = self.tensor.cores[node]
    if node.isleaf:
      if node.dim == self.phys_dim:
        return core, ['p', self._rank_label(node)]
      else:
        return core[self.indices[:, node.dim]], ['n', self._rank_label(node)]
    else:
      return core, [self._rank_label(c) for c in node.children] + [self._rank_label(node)]

  def _env_ops(self, node: TreeNode, E):
    """ All operands which together with the core of node contract to f (E is the environment of node). """
    return [E] + [self.U[c] for c in node.children]

  def update_up(self, node: TreeNode):
    """ Recompute the subtree contraction U[node] (children must be up to date). """
    ops = [self._core_op(node)] + [self.U[c] for c in node.children]
    out = _free_labels(ops) + [self._rank_label(node)]
    self.U[node] = (_contract(*ops, out=out), out)

  def env(self, node: TreeNode, E_parent=None):
    """ Environment of node, computed from the environment E_parent of its parent (siblings must be up to date). """
    if node.isroot:
      root_core = self.tensor.cores[node]
      return torch.ones(1, dtype=root_core.dtype, device=root_core.device), [self._rank_label(node)]
    parent = node.parent
    ops = [self._core_op(parent), E_parent] + [self.U[s] for s in parent.children if s.id != node.id]
    out = _free_labels(ops) + [self._rank_label(node)]
    return _contract(*ops, out=out), out

  def dfs(self, update_up: bool = False):
    """
    Depth-first traversal yielding (node, E) in pre-order, where E is the environment of node.
    Requires all U to be up to date.

    Environments are computed lazily when the generator is advanced, so the core of a yielded
    node may be modified before advancing. With update_up, the subtree contraction of every node
    is updated once its subtree is finished. The environment of a node is freed as soon as the
    environment of its last child is computed, since most environments carry the physical
    dimension (N x Nx x r). Consumers should not hold on to E beyond the loop body.
    """
    root = self.tree.root
    stack = [[root, self.env(root), 0]] # node, environment, index of next child
    yield root, stack[-1][1]
    while stack:
      frame = stack[-1]
      node, i = frame[0], frame[2]
      if i == node.n_children:
        stack.pop()
        if update_up: self.update_up(node)
        continue
      child = node.children[i]
      E = self.env(child, frame[1])
      frame[2] += 1
      if i == node.n_children - 1: frame[1] = None
      stack.append([child, E, 0])
      yield child, E
      del E

  def up_sweep(self, node: TreeNode = None):
    if node is None: node = self.tree.root
    for child in node.children:
      self.up_sweep(child)
    self.update_up(node)

  def forward(self, indices):
    self.set_indices(indices)
    self.up_sweep()
    return self.value()

  def value(self):
    """ Tensor evaluation (N, Nx) from the current root subtree contraction. """
    root = self.tree.root
    return _contract(self.U[root], self.env(root), out=['n', 'p'])

  def local_apply(self, node: TreeNode, core, E):
    """ f = A_node(core), linear in core. """
    return _contract(self._core_op(node, core), *self._env_ops(node, E), out=['n', 'p'])

  def local_grad(self, node: TreeNode, R, E):
    """ A_node^T R for R of shape (N, Nx). """
    ops = [(R, ['n', 'p'])] + self._env_ops(node, E)
    _, labels = self._core_op(node)
    if node.isleaf and node.dim != self.phys_dim:
      rows = _contract(*ops, out=labels)
      G = torch.zeros_like(self.tensor.cores[node])
      return G.index_add_(0, self.indices[:, node.dim], rows)
    else:
      return _contract(*ops, out=labels)


@dataclass(frozen=True)
class ARGS_tree_tensor_training_explicit:
  verbose : int = 0
  Nx : int = 100
  Ny : int = 10
  Ny_decay : bool = True
  n_param : int = 200
  c_offset : float = 1
  c_var : float = 4
  c_decay : float = 2.
  tree_type : str = 'balanced'
  tree_weight_decay : float = 0.
  N_train : int = int(1e4)
  N_test : int = int(1e3)
  optimizer : str = 'adam' # 'adam', 'sd' (steepest descent with exact line search) or 'lbfgs'
  lr_start : float = 1e-2
  lr_end : float = 1e-6
  rinit : int = 20
  epoch_outer : int = 100
  epoch_inner : int = 10
  eps : float = 1e-4
  rng_seed : int = 0
  N_mc : int = 1000
  device : str = 'cpu'

def optimize_core(model: ExplicitGradTreeTensor, node: TreeNode, E, target, args: ARGS_tree_tensor_training_explicit, lr: float):
  """
  Run args.epoch_inner gradient steps on the core of node (with environment E), all other cores fixed.
  Since f is linear in the core, only local contractions are needed per step.
  No values are moved to the host, so on GPU the steps run without synchronization.
  """
  core = model.tensor.cores[node]

  if args.optimizer == 'lbfgs':
    # one step runs up to epoch_inner iterations, lr schedule is not used.
    opt = optim.LBFGS([core], lr=1, max_iter=args.epoch_inner, line_search_fn='strong_wolfe')
    def closure():
      res = model.local_apply(node, core, E) - target
      # LBFGS flattens the gradient with view, einsum output may not be contiguous
      core.grad = model.local_grad(node, 2 * res / res.numel(), E).contiguous()
      return torch.mean(torch.square(res))
    opt.step(closure)
    core.grad = None
    return

  if args.optimizer == 'adam':
    opt = optim.Adam([core], lr=lr)

  f = model.local_apply(node, core, E)
  for _ in range(args.epoch_inner):
    res = f - target
    G = model.local_grad(node, 2 * res / res.numel(), E)

    if args.optimizer == 'adam':
      core.grad = G
      opt.step()
      f = model.local_apply(node, core, E)
    elif args.optimizer == 'sd':
      # exact line search: f(core - a*G) = f - a*A(G), step 0 if A(G) = 0
      AG = model.local_apply(node, G, E)
      AG_sq = torch.sum(AG * AG)
      a = torch.where(AG_sq > 0, torch.sum(res * AG) / AG_sq, 0.)
      core -= a * G
      f = f - a * AG
    else:
      raise ValueError(f'Unknown optimizer {args.optimizer}')

  core.grad = None

def sweep(model: ExplicitGradTreeTensor, target, args: ARGS_tree_tensor_training_explicit, lr: float):
  """
  Depth-first sweep: optimize node, then descend into children. Environments are
  passed down and subtree contractions are updated on the way up, so every
  core is optimized with all other cores at their latest values.

  Requires all U to be up to date.
  """
  for node, E in model.dfs(update_up=True):
    if args.verbose: print('\r' + f'core.{node.id}', end='')
    optimize_core(model, node, E, target, args, lr)
    del E

def tree_tensor_training_explicit(args: ARGS_tree_tensor_training_explicit, return_tensor: bool = False):
  print('Running tree tensor training (explicit gradients)...')
  if args.verbose: print(args)

  stats = {'train_loss': [], 'test_loss': [], 'size': [], 'lr': []}
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  cfun = lambda x : torch.from_numpy(cfun_np(x))

  rng = np.random.default_rng(seed=args.rng_seed)

  if args.Ny_decay:
    tensor_shape = np.ceil(args.Ny *  np.log(2) / np.log(np.arange(2, args.n_param+2)))
    tensor_shape = tensor_shape[tensor_shape > 1]
    tensor_shape = tuple(int(ny) for ny in tensor_shape)
  else:
    tensor_shape = tuple([args.Ny] * args.n_param)

  tensor_shape = (args.Nx,) + tensor_shape

  tree_kwargs = {
    'weighted': dict(weights=(np.arange(args.n_param)+1)**-args.tree_weight_decay),
    'random': dict(seed=args.rng_seed),
  }.get(args.tree_type, {})
  tree = Tree.from_tupletree(zero_root_tupletree(args.n_param, args.tree_type, **tree_kwargs))

  stats['sacking_index'] = np.sum(np.array(tree.get_leaf_depths()[1:])-1)
  colles_ind = 0
  subtree_dims = tree.get_subtree_dims()
  for node in tree.node_list:
    if node.isroot: continue
    if not node.isleaf and node.n_children == 2:
      colles_ind += abs(len(subtree_dims[node.children[0]]) - len(subtree_dims[node.children[1]]))

  stats['colles_index'] = colles_ind

  train_ind = []
  train_eval = []
  test_ind = []
  test_eval = []
  for i in range(args.N_train):
    y = rng.integers(0, tensor_shape)
    train_ind += [torch.from_numpy(y)]
    train_eval += [cfun(np.hstack([np.arange(args.Nx).reshape((-1,1)), np.tile(y[1:], [args.Nx,1])]))]
  train_ind = torch.vstack(train_ind).to(args.device)
  train_eval = torch.vstack(train_eval).to(args.device)

  for i in range(args.N_test):
    y = rng.integers(0, tensor_shape)
    test_ind += [torch.from_numpy(y)]
    test_eval += [cfun(np.hstack([np.arange(args.Nx).reshape((-1,1)), np.tile(y[1:], [args.Nx,1])]))]
  test_ind = torch.vstack(test_ind).to(args.device)
  test_eval = torch.vstack(test_eval).to(args.device)

  tensor = TreeBasedTensor.randn(tree, tensor_shape, args.rinit, seed=args.rng_seed, device=args.device)
  model = ExplicitGradTreeTensor(tensor)

  lr = args.lr_start

  with torch.no_grad():
    for epoch in range(args.epoch_outer):
      model.set_indices(train_ind)
      model.up_sweep()
      sweep(model, train_eval, args, lr)
      stats['train_loss'] += [torch.nn.functional.mse_loss(model.value(), train_eval).item()]

      model.tensor.round(args.eps)
      stats['size'] += [model.tensor.sparse_size]
      stats['lr'] += [lr]

      eval = model.forward(test_ind)
      stats['test_loss'] += [torch.nn.functional.mse_loss(eval, test_eval).item()]

      if args.verbose:
        print('\r' + f'epoch {epoch}, train loss {stats['train_loss'][-1]:.2e}, validation loss {stats['test_loss'][-1]:.2e}, lr {lr:.1e}, size {model.tensor.sparse_size}')

      if epoch > 5 and sum(stats['test_loss'][-4:]) / 4 > 1.0 * min(stats['test_loss'][:-4]):
        lr *= .5
      if lr < args.lr_end:
        break

    # coeff TT error estimate
    errs = coeff_mc_errors(model.tensor, cfun, rng, args.N_mc)

  stats['MC_errors'] = errs
  if args.verbose:  print(f'Coefficient TT RMS l2 err = {RMS(errs):.2e}')

  fn = f'data/stats_explicit_{hash(args):X}.json'
  stats['file'] = fn
  with open(fn, mode='w') as file:
    json.dump({'args': asdict(args), 'stats':stats}, file, cls=NumpyEncoder)

  if args.verbose: print(f'\nSaved stats to: {fn}')

  if return_tensor:
    return stats, model.tensor
  return stats
