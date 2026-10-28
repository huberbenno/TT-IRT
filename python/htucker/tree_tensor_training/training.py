"""
Tree tensor training: fit a tree tensor to samples of the coefficient with core-wise sweeps.

The tensor is linear in each core, so the gradient and the local least squares problem of a core
only need contractions with its environment (ExplicitGradTreeTensor). The physical dimension can be
reduced to the rank of its leaf (reduced_sweep), and the local least squares problems are solved
with preconditioned CGLS (lstsq_core).
"""
import time
import warnings
import numpy as np
from dataclasses import dataclass

import torch
from torch import optim
from tree.tree import Tree, TreeNode, NodeIndexedList
from tree.tree_util import zero_root_tupletree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree_cross.tree_cross_torch import TreeCross
from pde.coeff_function import coeff
from benchmarks.util import RMS, coeff_mc_errors


@dataclass(frozen=True)
class ARGS_tree_tensor_training:
  verbose : int = 0 # 1: one line per epoch, 2: also the core currently optimized
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
  optimizer : str = 'lstsq' # 'adam', 'lbfgs', 'sd' (steepest descent with exact line search) or 'lstsq' (per-core least squares by CGLS)
  lr_start : float = 1e-2 # 'lstsq', 'sd' and 'lbfgs' do not use the lr, the schedule then only decides when to stop
  lr_end : float = 1e-3 # training stops once lr is below (1e-3: about 4 * lr_patience epochs without improvement)
  lr_factor : float = .5 # lr is reduced by this factor when the test loss has not improved
  lr_patience : int = 5 # for this many epochs
  lr_threshold : float = 1e-3 # by more than this relative amount
  init : str = 'mean' # 'uniform': random with rank rinit, 'cross': random with the ranks of a tree cross approximation,
                         # 'cross_tensor': the cross approximation itself, perturbed by init_noise,
                         # 'mean': constant at the mean of the training data plus a random tensor with the cross ranks
  init_noise : float = 1e-2 # 'cross_tensor': relative standard deviation of the perturbation of every core,
                            # 'mean': standard deviation of the random entries relative to the mean
  rinit : int = 20
  epoch_outer : int = 100
  epoch_inner : int = 2 # optimizer steps (CGLS iterations for 'lstsq') per core and sweep
  rounding : bool = False # round the tensor during and after training (to extra_eps * eps and eps)
  eps : float = 1e-4 # rounding tolerance of the trained tensor
  lstsq_precondition : bool = True # 'lstsq': preconditioned CGLS, exact block preconditioner for parameter leaves, Kronecker approximation otherwise
  phys_reduction : bool = True # fit all but the physical leaf to the target projected onto its range (see reduced_sweep)
  round_start : int = 0 # with rounding: first epoch after which the tensor is rounded (the final rounding to eps is always done)
  extra_eps : float = .1 # during training the tensor is rounded to extra_eps * eps, so ranks are not cut too early
  cross_eps : float = 1e-4
  cross_niter : int = 15
  cross_kickrank : int = 5
  rng_seed : int = 0
  N_mc : int = 1000
  device : str = 'cpu'

def synchronize(device):
  device = torch.device(device)
  if device.type == 'cuda': torch.cuda.synchronize(device)

def tensor_shape(args: ARGS_tree_tensor_training):
  """ (Nx,) + parameter dimensions. """
  if args.Ny_decay:
    param_shape = np.ceil(args.Ny *  np.log(2) / np.log(np.arange(2, args.n_param+2)))
    param_shape = param_shape[param_shape > 1]
    param_shape = tuple(int(ny) for ny in param_shape)
  else:
    param_shape = tuple([args.Ny] * args.n_param)
  return (args.Nx,) + param_shape

def build_tree(args: ARGS_tree_tensor_training) -> Tree:
  tree_kwargs = {
    'weighted': dict(weights=(np.arange(args.n_param)+1)**-args.tree_weight_decay),
    'random': dict(seed=args.rng_seed),
  }.get(args.tree_type, {})
  return Tree.from_tupletree(zero_root_tupletree(args.n_param, args.tree_type, **tree_kwargs))

def tree_indices(tree: Tree):
  """ Sackin and Colless index of the tree (the root edge to dimension 0 excluded). """
  stats = {}
  stats['sacking_index'] = np.sum(np.array(tree.get_leaf_depths()[1:])-1)
  colles_ind = 0
  subtree_dims = tree.get_subtree_dims()
  for node in tree.node_list:
    if node.isroot: continue
    if not node.isleaf and node.n_children == 2:
      colles_ind += abs(len(subtree_dims[node.children[0]]) - len(subtree_dims[node.children[1]]))
  stats['colles_index'] = colles_ind
  return stats

def sample_data(cfun, shape, N, rng, device):
  """ N random indices (N, ndim) and the coefficient along the physical dimension at each of them (N, Nx). """
  Nx = shape[0]
  ind = []
  vals = []
  for i in range(N):
    y = rng.integers(0, shape)
    ind += [torch.from_numpy(y)]
    vals += [cfun(np.hstack([np.arange(Nx).reshape((-1,1)), np.tile(y[1:], [Nx,1])]))]
  return torch.vstack(ind).to(device), torch.vstack(vals).to(device)

CROSS_WARN_FACTOR = 10 # warn if the relative error of the cross approximation exceeds this factor times cross_eps

def cross_approximation(cfun, tree: Tree, shape, args: ARGS_tree_tensor_training):
  """ Tree cross approximation of the coefficient, rounded to args.cross_eps. Returns the tensor and the number of evaluations. """
  C = TreeBasedTensor.randn(tree, shape, args.Ny, seed=args.rng_seed)
  cross = TreeCross(C, seed=args.rng_seed)
  cross.run(cfun, n_iter=args.cross_niter, eps=args.cross_eps, kickrank=args.cross_kickrank, verbose=args.verbose > 1)
  return C.round(args.cross_eps), cross.n_eval

def perturb(tensor: TreeBasedTensor, noise: float, seed=None, device=None) -> TreeBasedTensor:
  """ Copy of tensor with Gaussian noise added to every core, with standard deviation noise times the RMS of the core entries. """
  rng = torch.Generator()
  if seed is not None: rng.manual_seed(seed)
  cores = []
  for core in tensor.cores:
    scale = noise * torch.sqrt(torch.mean(core**2))
    cores += [core + scale * torch.randn(core.shape, generator=rng, dtype=core.dtype)]
  return TreeBasedTensor(cores, tensor.tree, device=device)

class LRSchedule:
  """
  Reduce the learning rate on a plateau: lr is multiplied by factor once the loss has not improved
  on its best value by more than the relative threshold for patience epochs in a row. The count
  restarts after every reduction, so consecutive reductions are at least patience epochs apart.
  """
  def __init__(self, lr: float, factor: float = .5, patience: int = 5, threshold: float = 1e-3) -> None:
    self.lr = lr
    self.factor = factor
    self.patience = patience
    self.threshold = threshold
    self.best = np.inf
    self.n_bad = 0

  def step(self, loss: float) -> float:
    if loss < self.best * (1 - self.threshold):
      self.best = loss
      self.n_bad = 0
    else:
      self.n_bad += 1
      if self.n_bad >= self.patience:
        self.lr *= self.factor
        self.n_bad = 0
    return self.lr


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


PRECOND_JITTER = 1e-8 # relative regularization of the (factors of the) local Gram matrices
CG_RTOL = 1e-12 # CGLS steps stop once the residual of the normal equations decreased by this factor

def _self_gram(op, label):
  """ Gram matrix of the einsum operand op = (tensor, labels) over all its labels but label (r x r). """
  t, labels = op
  copy = [('copy', l) if l == label else l for l in labels]
  return _contract(op, (t, copy), out=[label, ('copy', label)])

def _inv_cholesky(C, jitter=PRECOND_JITTER):
  """ L^-1 for the Cholesky factor L of C (batched), regularized relative to the trace (identity for C = 0). """
  r = C.shape[-1]
  eye = torch.eye(r, dtype=C.dtype, device=C.device)
  delta = jitter * torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / r
  delta = torch.where(delta > 0, delta, 1.)
  L, _ = torch.linalg.cholesky_ex(C + delta[..., None, None] * eye)
  return torch.linalg.solve_triangular(L, eye.expand(L.shape), upper=False)

def kron_gram_factors(model: ExplicitGradTreeTensor, node: TreeNode, E):
  """
  Factors of the Kronecker approximation of the local Gram matrix A^T A of node,
    A^T A = sum_n kron(u_1 u_1^T, ..., E_n^T E_n) ~ kron(C_1, ..., C_E),
  with C_k the Gram matrix of the operand (child contraction or environment) contracted with core mode k.
  Returns one C_k per core mode, None for modes no operand contracts with (the physical mode of the
  physical leaf, whose factor is the identity). For the physical leaf the approximation is exact.
  """
  ops = model._env_ops(node, E)
  _, labels = model._core_op(node)
  grams = []
  for label in labels:
    op = next((op for op in ops if label in op[1]), None)
    grams += [None if op is None else _self_gram(op, label)]
  return grams

def leaf_gram_blocks(model: ExplicitGradTreeTensor, node: TreeNode, E):
  """
  Exact local Gram matrix A^T A of a parameter leaf: every row of the core is only used by the samples
  with that index value, so A^T A is block diagonal with one r x r block per index value (Ny x r x r).
  """
  label = model._rank_label(node)
  t, labels = E
  copy = [('copy', l) if l == label else l for l in labels]
  out = [label, ('copy', label)]
  if 'n' in labels:
    EE = _contract(E, (t, copy), out=['n'] + out)
  else: # the same environment for all samples
    EE = _contract(E, (t, copy), out=out).expand(model.indices.shape[0], -1, -1)
  core = model.tensor.cores[node]
  G = torch.zeros(core.shape[:1] + EE.shape[1:], dtype=EE.dtype, device=EE.device)
  return G.index_add_(0, model.indices[:, node.dim], EE)

def _apply_modewise(core, mats, transpose: bool = False):
  """ Multiply every mode k of core by mats[k] (or its transpose), modes with None are left unchanged. """
  for k, M in enumerate(mats):
    if M is None: continue
    if transpose: M = M.T
    core = torch.movedim(torch.tensordot(M, core, dims=([1], [k])), 0, k)
  return core

def lstsq_core(model: ExplicitGradTreeTensor, node: TreeNode, E, target, n_iter: int, precondition: bool = False):
  """
  Approximately solve the least squares problem min ||A(core) - target|| for the core of node
  (all other cores fixed) with n_iter iterations of CGLS, i.e. CG on the normal equations
  A^T A core = A^T target, starting from the current core. A and A^T are only applied through
  local contractions, A^T A is never formed. Steps are 0 once the residual of the normal
  equations decreased by CG_RTOL, so no values are moved to the host.

  With precondition, CGLS runs on A S in z with core = S z, S = L^-T for a factorization L L^T of
  (an approximation of) A^T A: the exact block diagonal one for parameter leaves (leaf_gram_blocks),
  the Kronecker approximation otherwise (kron_gram_factors), which is exact for the physical leaf.
  """
  if not precondition:
    S = St = lambda x: x
  elif node.isleaf and node.dim != model.phys_dim:
    Linv = _inv_cholesky(leaf_gram_blocks(model, node, E))
    S = lambda z: torch.einsum('ikj,ik->ij', Linv, z) # row i: L_i^-T z_i
    St = lambda g: torch.einsum('ijk,ik->ij', Linv, g)
  else:
    Linv = [None if C is None else _inv_cholesky(C) for C in kron_gram_factors(model, node, E)]
    S = lambda z: _apply_modewise(z, Linv, transpose=True)
    St = lambda g: _apply_modewise(g, Linv)
  core = model.tensor.cores[node]
  R = target - model.local_apply(node, core, E) # residual
  G = St(model.local_grad(node, R, E)) # residual of the (preconditioned) normal equations
  P = G
  gamma = torch.sum(G * G)
  gamma_stop = CG_RTOL**2 * gamma
  for _ in range(n_iter):
    active = gamma > gamma_stop
    SP = S(P)
    Q = model.local_apply(node, SP, E)
    QQ = torch.sum(Q * Q)
    alpha = torch.where(active & (QQ > 0), gamma / QQ, 0.)
    core += alpha * SP
    R = R - alpha * Q
    G = St(model.local_grad(node, R, E))
    gamma_new = torch.sum(G * G)
    beta = torch.where(active & (gamma > 0), gamma_new / gamma, 0.)
    P = G + beta * P
    gamma = gamma_new

def optimize_core(model: ExplicitGradTreeTensor, node: TreeNode, E, target, args: ARGS_tree_tensor_training, lr: float,
                  n_values: int = None):
  """
  Run args.epoch_inner gradient steps on the core of node (with environment E), all other cores fixed.
  Since f is linear in the core, only local contractions are needed per step.
  No values are moved to the host, so on GPU the steps run without synchronization.
  The loss is the sum of squared residuals divided by n_values (default: the number of target values,
  i.e. the MSE), the reduced problem of reduced_sweep passes the size of the full target.
  """
  core = model.tensor.cores[node]
  n = target.numel() if n_values is None else n_values

  if args.optimizer == 'lbfgs':
    # one step runs up to epoch_inner iterations, lr schedule is not used.
    opt = optim.LBFGS([core], lr=1, max_iter=args.epoch_inner, line_search_fn='strong_wolfe')
    def closure():
      res = model.local_apply(node, core, E) - target
      # LBFGS flattens the gradient with view, einsum output may not be contiguous
      core.grad = model.local_grad(node, 2 * res / n, E).contiguous()
      return torch.sum(torch.square(res)) / n
    opt.step(closure)
    core.grad = None
    return

  if args.optimizer == 'lstsq':
    lstsq_core(model, node, E, target, args.epoch_inner, precondition=args.lstsq_precondition)
    return

  if args.optimizer == 'adam':
    opt = optim.Adam([core], lr=lr)

  f = model.local_apply(node, core, E)
  for _ in range(args.epoch_inner):
    res = f - target
    G = model.local_grad(node, 2 * res / n, E)

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

def sweep(model: ExplicitGradTreeTensor, target, args: ARGS_tree_tensor_training, lr: float,
          skip: TreeNode = None, n_values: int = None):
  """
  Depth-first sweep: optimize node, then descend into children. Environments are
  passed down and subtree contractions are updated on the way up, so every
  core is optimized with all other cores at their latest values. The core of skip is not optimized.

  Requires all U to be up to date.
  """
  for node, E in model.dfs(update_up=True):
    if skip is None or node.id != skip.id:
      if args.verbose > 1: print('\r' + f'core.{node.id}', end='')
      optimize_core(model, node, E, target, args, lr, n_values)
    del E

def orthonormalize_phys_leaf(model: ExplicitGradTreeTensor):
  """
  QR of the physical leaf core P = Q R, R is absorbed into the parent core, so the tensor is unchanged.
  Returns the leaf and Q, which is its new core.
  """
  leaf = model.tree.dim2leaf(model.phys_dim)
  Q, R = torch.linalg.qr(model.tensor.cores[leaf])
  i = leaf.child_ind
  parent_core = model.tensor.cores[leaf.parent]
  model.tensor.cores[leaf.parent] = torch.movedim(torch.tensordot(R, parent_core, dims=([1], [i])), 0, i)
  model.tensor.cores[leaf] = Q
  return leaf, Q

def reduced_sweep(model: ExplicitGradTreeTensor, target, args: ARGS_tree_tensor_training, lr: float):
  """
  Sweep with the physical dimension (size Nx) reduced to the rank r0 of the physical leaf.

  With the physical leaf core Q orthonormal, f = Q g where g (N x r0) contracts all other cores, and
    ||f - y||^2 = ||g - y Q||^2 + ||y||^2 - ||y Q||^2.
  So all other cores are fitted to the reduced target y Q with the leaf core replaced by the identity,
  their gradients are exactly those of the full problem. Environments then carry r0 instead of Nx.
  Finally the physical leaf is optimized on the full target, its environment g has no physical index.
  The leaf is updated last instead of at its position in the depth-first order of sweep.

  Requires the indices to be set (set_indices).
  """
  leaf, Q = orthonormalize_phys_leaf(model)
  model.tensor.cores[leaf] = torch.eye(Q.shape[1], dtype=Q.dtype, device=Q.device)
  model.up_sweep()
  sweep(model, target @ Q, args, lr, skip=leaf, n_values=target.numel())
  model.tensor.cores[leaf] = Q

  # environment of the leaf from the root down, the subtree contractions off the path are up to date
  path = [leaf]
  while not path[-1].isroot: path.append(path[-1].parent)
  E = None
  for node in reversed(path):
    E = model.env(node, E)
  optimize_core(model, leaf, E, target, args, lr)



class Trainer:
  """ Model and settings of the training, sweeps and evaluation at given indices. """
  def __init__(self, tensor: TreeBasedTensor, args: ARGS_tree_tensor_training) -> None:
    self.model = ExplicitGradTreeTensor(tensor)
    self.args = args

  @property
  def tensor(self) -> TreeBasedTensor:
    return self.model.tensor

  def sweep(self, ind, target, lr: float):
    with torch.no_grad():
      self.model.set_indices(ind)
      leaf_core = self.tensor.cores[self.model.tree.dim2leaf(self.model.phys_dim)]
      # without reduction if it would not reduce the physical dimension (and change the rank of the leaf)
      if self.args.phys_reduction and leaf_core.shape[1] < leaf_core.shape[0]:
        reduced_sweep(self.model, target, self.args, lr)
      else:
        self.model.up_sweep()
        sweep(self.model, target, self.args, lr)

  def forward(self, ind):
    with torch.no_grad():
      return self.model.forward(ind)


def train(args: ARGS_tree_tensor_training, return_tensor: bool = False):
  """
  Fit a tree tensor to samples of the coefficient with core-wise sweeps, optionally followed by rounding.

  Returns the stats: per epoch losses, sizes and timings of the phases sweep, round and eval
  (t_elapsed excludes eval), and after training the Monte Carlo coefficient errors.
  Train and test loss are the MSE over the full data sets after the epoch (with rounding: after rounding
  to extra_eps * eps), the final ones (final_train_loss, final_test_loss) after the final rounding to eps.
  """
  if args.verbose: print('Running tree tensor training...')
  if args.verbose: print(args)
  device = torch.device(args.device)

  stats = {key: [] for key in ['train_loss', 'test_loss', 'size', 'max_rank', 'lr', 't_sweep', 't_round', 't_eval', 't_elapsed']}
  t = time.perf_counter()
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  cfun = lambda x : torch.from_numpy(cfun_np(x))

  rng = np.random.default_rng(seed=args.rng_seed)
  shape = tensor_shape(args)
  tree = build_tree(args)
  stats.update(tree_indices(tree))

  # MC errors of the cross and the trained tensor at the same parameter samples
  mc_errors = lambda tensor: coeff_mc_errors(tensor, cfun, np.random.default_rng(seed=args.rng_seed + 1), args.N_mc)
  if args.init in ('cross', 'cross_tensor', 'mean'):
    t_cross = time.perf_counter()
    C, stats['cross_neval'] = cross_approximation(cfun, tree, shape, args)
    stats['t_cross'] = time.perf_counter() - t_cross
    ranks = [core.shape[-1] for core in C.cores]
    stats['cross_size'] = C.sparse_size
    stats['cross_ranks'] = ranks
    errs = mc_errors(C)
    stats['cross_MC_rms'] = RMS(errs)
    stats['cross_MC_max'] = np.max(errs)
    if args.verbose: print(f'Cross approximation: RMS l2 err = {stats["cross_MC_rms"]:.2e}, size {C.sparse_size}, max rank {max(ranks)}, {stats["t_cross"]:.1f}s')
  elif args.init == 'uniform':
    ranks = args.rinit
  else:
    raise ValueError(f'Unknown init {args.init}')

  t_data = time.perf_counter()
  train_ind, train_eval = sample_data(cfun, shape, args.N_train, rng, device)
  test_ind, test_eval = sample_data(cfun, shape, args.N_test, rng, device)
  stats['t_data'] = time.perf_counter() - t_data

  if 'cross_MC_rms' in stats:
    # relative to the RMS of the coefficient, estimated from the training data
    stats['cross_rel_err'] = stats['cross_MC_rms'] / torch.sqrt(torch.mean(train_eval**2)).item()
    if stats['cross_rel_err'] > CROSS_WARN_FACTOR * args.cross_eps:
      warnings.warn(f'Relative error {stats["cross_rel_err"]:.1e} of the cross approximation exceeds '
                    f'{CROSS_WARN_FACTOR} x cross_eps = {CROSS_WARN_FACTOR * args.cross_eps:.1e}, '
                    f'consider more sweeps (cross_niter = {args.cross_niter}).')

  if args.init == 'cross_tensor':
    tensor = perturb(C, args.init_noise, seed=args.rng_seed, device=device)
  elif args.init == 'mean':
    # rank 1 constant plus random part, the ranks are the cross ranks + 1 (until the first rounding, if any)
    mean = train_eval.mean().item()
    tensor = mean * TreeBasedTensor.ones(tree, shape) \
      + TreeBasedTensor.randn(tree, shape, ranks, seed=args.rng_seed, scale=args.init_noise * abs(mean))
    tensor = TreeBasedTensor(tensor.cores, tree, device=device)
  else:
    tensor = TreeBasedTensor.randn(tree, shape, ranks, seed=args.rng_seed, device=device)
  if args.init != 'uniform': del C
  stats['init_size'] = tensor.sparse_size
  trainer = Trainer(tensor, args)
  synchronize(device)
  stats['t_setup'] = time.perf_counter() - t
  if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)

  def timed(fun):
    synchronize(device)
    t = time.perf_counter()
    res = fun()
    synchronize(device)
    return res, time.perf_counter() - t

  schedule = LRSchedule(args.lr_start, args.lr_factor, args.lr_patience, args.lr_threshold)
  lr = schedule.lr
  def evaluate():
    with torch.no_grad():
      train_loss = torch.nn.functional.mse_loss(trainer.forward(train_ind), train_eval).item()
      test_loss = torch.nn.functional.mse_loss(trainer.forward(test_ind), test_eval).item()
    return train_loss, test_loss

  t_elapsed = 0.
  for epoch in range(args.epoch_outer):
    _, t_sweep = timed(lambda: trainer.sweep(train_ind, train_eval, lr))
    t_round = 0.
    if args.rounding and epoch >= args.round_start:
      _, t_round = timed(lambda: trainer.tensor.round(args.extra_eps * args.eps))
    (train_loss, test_loss), t_eval = timed(evaluate)
    t_elapsed += t_sweep + t_round

    stats['train_loss'] += [train_loss]
    stats['test_loss'] += [test_loss]
    stats['size'] += [trainer.tensor.sparse_size]
    stats['max_rank'] += [max(core.shape[-1] for core in trainer.tensor.cores)]
    stats['lr'] += [lr]
    stats['t_sweep'] += [t_sweep]
    stats['t_round'] += [t_round]
    stats['t_eval'] += [t_eval]
    stats['t_elapsed'] += [t_elapsed]

    if args.verbose:
      print('\r' + f'epoch {epoch}, train loss {train_loss:.2e}, validation loss {test_loss:.2e}, lr {lr:.1e}, '
            f'size {trainer.tensor.sparse_size}, sweep {t_sweep:.2f}s', flush=True)

    lr = schedule.step(test_loss)
    if lr < args.lr_end:
      break

  stats['n_epochs'] = len(stats['train_loss'])
  if args.rounding:
    # final rounding to the target tolerance
    _, t_round = timed(lambda: trainer.tensor.round(args.eps))
    t_elapsed += t_round
  stats['final_train_loss'], stats['final_test_loss'] = evaluate()
  stats['t_train'] = t_elapsed
  stats['final_size'] = trainer.tensor.sparse_size
  stats['final_ranks'] = [core.shape[-1] for core in trainer.tensor.cores]
  if device.type == 'cuda': stats['cuda_peak_MB'] = torch.cuda.max_memory_allocated(device) / 1024**2

  # coeff tree tensor error estimate
  errs = mc_errors(trainer.tensor)
  stats['MC_errors'] = errs
  stats['MC_rms'] = RMS(errs)
  stats['MC_max'] = np.max(errs)
  if args.verbose: print(f'Coefficient tree tensor RMS l2 err = {stats["MC_rms"]:.2e}, training time {t_elapsed:.2f}s')

  if return_tensor:
    return stats, trainer.tensor
  return stats
