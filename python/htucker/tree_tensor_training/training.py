"""
Tree tensor training: fit a tree tensor to samples of the coefficient (target 'coeff') or of the
solution of the parametric 1d diffusion problem (target 'diffusion') with core-wise sweeps.

The tensor is linear in each core, so the gradient and the local least squares problem of a core
only need contractions with its environment (ExplicitGradTreeTensor). The physical dimension can be
reduced to the rank of its leaf (reduced_sweep), and the local least squares problems are solved
with preconditioned CGLS (lstsq_core). The ranks are given, rounded during or after training, or grown
from rank 1 by rank adaptation (rank_adapt).
"""
import time
import warnings
import numpy as np
from dataclasses import dataclass, replace

import torch
from torch import optim
from tree.tree import Tree, TreeNode, NodeIndexedList
from tree.tree_util import zero_root_tupletree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree_cross.tree_cross_torch import TreeCross
from pde.coeff_function import coeff
from pde.diffusion_1d import Diffusion1D_torch
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
  target : str = 'coeff' # 'coeff': the coefficient (Nx values per sample),
                         # 'diffusion': the PDE solution (Nx+1 values per sample)
  N_train : int = int(1e4)
  N_test : int = int(1e3) # validation set (lr schedule / stopping), ignored with val_fraction
  val_fraction : float = 0. # > 0: N_train is the total sample budget,
                            # of which max(2, round(val_fraction * N_train)) validate
  ref_file : str = None # torch file with a reference set {'ind', 'vals'} for the MC error
                        # (default: N_mc fresh samples)
  optimizer : str = 'lstsq' # 'adam', 'lbfgs', 'sd' (steepest descent with exact line search)
                            # or 'lstsq' (per-core least squares by CGLS)
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
  init_ranks : tuple = None # ranks by node id for init 'uniform' and 'mean'
                            # (instead of rinit / the cross ranks, no cross is run)
  epoch_outer : int = 100
  epoch_inner : int = 2 # optimizer steps (CGLS iterations for 'lstsq') per core and sweep
  rank_adapt : bool = False # grow the ranks from rank 1 (Grelier, Nouy, Chevreuil 2019, Sec. 4.1),
                            # init / init_ranks ignored
  adapt_theta : float = .8 # nodes with eta >= theta * max(eta) get rank + 1,
                           # theta <= adapt_theta (the largest admissible)
  adapt_iter : int = 30 # maximal number of rank increases
  adapt_epochs : int = 20 # epoch limit of every fit at fixed ranks (warm started)
  adapt_overfit : float = 10. # stop once the validation loss exceeds this factor times the best one
  adapt_patience : int = 5 # stop after this many rank increases without a better validation loss (at small N it
                           # often creeps for 10-25 rank increases before it drops 3-5x, so any improvement counts)
  adapt_init : str = 'enrich' # start of the fit at the new ranks: 'enrich': u with the selected edges padded by
                              # small random entries (enrich), 'truncate': u~ truncated to the new ranks
  adapt_select_ridge : float = 0. # ridge of the fit of u~, which only selects the nodes (None: ridge); with
                                  # ridge > 0 there the new directions of u~ vanish and no rank grows
  orthogonalize : bool = False # orthogonal form of the tensor at the start of every sweep (resets the gauge);
                                # in tests (d = 50) slower convergence and errors of O(1) at a few parameter
                                # points: leaf rows of rarely sampled index values are mixed by every QR
  keep_best : bool = False # return the tensor with the best validation loss of every training phase,
                           # its starting tensor included
  track_ref : bool = False # record the relative MC error (as MC_rel_rms) after every epoch (ref_rel_err)
                           # and of the starting tensor of every phase (start_ref_rel_err)
  rounding : bool = False # round the tensor during and after training (to extra_eps * eps and eps)
  round_mode : str = 'fixed' # with rounding, 'fixed': round every epoch to extra_eps * eps and finally to eps,
                             # 'val': train at full ranks until the schedule stops, round to the tolerance in
                             # {0} u {eps * 10^(k/2)} <= 1 with the best validation loss, then train again
  eps : float = 1e-4 # rounding tolerance of the trained tensor ('val': the smallest nonzero candidate)
  ridge : float = 0. # 'lstsq': Tikhonov regularization of every core solve, relative to the mean
                     # eigenvalue of the local Gram matrix (see lstsq_core)
  lstsq_precondition : bool = True # 'lstsq': preconditioned CGLS, exact block preconditioner for
                                   # parameter leaves, Kronecker approximation otherwise
  phys_reduction : bool = True # fit all but the physical leaf to the target projected onto its range
                               # (see reduced_sweep)
  round_start : int = 0 # with rounding: first epoch after which the tensor is rounded
                        # (the final rounding to eps is always done)
  extra_eps : float = .1 # during training the tensor is rounded to extra_eps * eps,
                         # so ranks are not cut too early
  cross_eps : float = 1e-4
  cross_niter : int = 15
  cross_kickrank : int = 5
  rng_seed : int = 0
  N_mc : int = 1000
  device : str = 'cpu'

INITS = ('uniform', 'cross', 'cross_tensor', 'mean')
ROUND_MODES = ('fixed', 'val')
ADAPT_INITS = ('enrich', 'truncate')

def synchronize(device):
  device = torch.device(device)
  if device.type == 'cuda': torch.cuda.synchronize(device)

def ranks_of(tensor: TreeBasedTensor):
  """ Ranks by node id (the root's is 1). """
  return [core.shape[-1] for core in tensor.cores]

def coeff_shape(args: ARGS_tree_tensor_training):
  """ (Nx,) + parameter dimensions. """
  if args.Ny_decay:
    param_shape = np.ceil(args.Ny *  np.log(2) / np.log(np.arange(2, args.n_param+2)))
    param_shape = param_shape[param_shape > 1]
    param_shape = tuple(int(ny) for ny in param_shape)
  else:
    param_shape = tuple([args.Ny] * args.n_param)
  return (args.Nx,) + param_shape

def tensor_shape(args: ARGS_tree_tensor_training):
  """ Shape of the target: (n_out,) + parameter dimensions, n_out = Nx (coefficient) or Nx+1 (solution nodes). """
  shape = coeff_shape(args)
  if args.target == 'coeff': return shape
  if args.target == 'diffusion': return (args.Nx + 1,) + shape[1:]
  raise ValueError(f'Unknown target {args.target}')

def coeff_values(cfun_np, Nx):
  """ Coefficient at the Nx grid points for a batch of parameter indices Y (N, d), as (N, Nx). """
  def values(Y):
    X = np.hstack([np.tile(np.arange(Nx), len(Y)).reshape((-1, 1)), np.repeat(Y, Nx, axis=0)])
    return torch.from_numpy(cfun_np(X).reshape((len(Y), Nx)))
  return values

def diffusion_values(cfun_np, Nx):
  """ P1 solution (Nx+1 nodes) of the diffusion problem with the coefficient on the Nx cells, as (N, Nx+1). """
  coeff_fun = coeff_values(cfun_np, Nx)
  pde = Diffusion1D_torch(Nx)
  def values(Y):
    C = coeff_fun(Y).T
    zeros = torch.zeros((1, len(Y)), dtype=C.dtype)
    return torch.hstack(pde.solve([C, zeros], [zeros])).T
  return values

def target_values(args: ARGS_tree_tensor_training):
  """ Function mapping parameter indices Y (N, d) to the target values (N, n_out). """
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  if args.target == 'coeff': return coeff_values(cfun_np, args.Nx)
  if args.target == 'diffusion': return diffusion_values(cfun_np, args.Nx)
  raise ValueError(f'Unknown target {args.target}')

def build_tree(args: ARGS_tree_tensor_training) -> Tree:
  tree_kwargs = {
    'weighted': dict(weights=(np.arange(args.n_param)+1)**-args.tree_weight_decay),
    'random': dict(seed=args.rng_seed),
  }.get(args.tree_type, {})
  return Tree.from_tupletree(zero_root_tupletree(args.n_param, args.tree_type, **tree_kwargs))

def tree_indices(tree: Tree):
  """ Sackin and Colless index of the tree (the root edge to dimension 0 excluded). """
  stats = {}
  stats['sackin_index'] = np.sum(np.array(tree.get_leaf_depths()[1:])-1)
  colless_ind = 0
  subtree_dims = tree.get_subtree_dims()
  for node in tree.node_list:
    if node.isroot: continue
    if not node.isleaf and node.n_children == 2:
      colless_ind += abs(len(subtree_dims[node.children[0]]) - len(subtree_dims[node.children[1]]))
  stats['colless_index'] = colless_ind
  return stats

def sample_data(values, shape, N, rng, device):
  """
  N random indices (N, ndim) and the target along the physical dimension at each of them (N, n_out).
  values maps the parameter indices (N, ndim-1) to the target values (see target_values). The index
  of the physical dimension (column 0) is drawn as well but not used.
  """
  ind = np.array([rng.integers(0, shape) for _ in range(N)], dtype=np.int64).reshape((N, len(shape)))
  return torch.from_numpy(ind).to(device), values(ind[:, 1:]).to(device)

def ref_errors(tensor: TreeBasedTensor, ind, vals, chunk: int = 1000):
  """
  RMS errors over the physical dimension (dimension 0) at the indices ind (N, ndim),
  vals (N, n_out) are the exact values.
  """
  model = ExplicitGradTreeTensor(tensor)
  errs = []
  with torch.no_grad():
    for i in range(0, len(ind), chunk):
      res = model.forward(ind[i:i+chunk].to(tensor.device)) - vals[i:i+chunk].to(tensor.device)
      errs += [torch.sqrt(torch.mean(res**2, dim=1))]
  return torch.cat(errs).numpy(force=True)

CROSS_WARN_FACTOR = 10 # warn if the relative error of the cross approximation exceeds this factor times cross_eps

def cross_approximation(cfun, tree: Tree, shape, args: ARGS_tree_tensor_training):
  """
  Tree cross approximation of the coefficient (shape (Nx,) + parameter dimensions),
  rounded to args.cross_eps. Returns the tensor and the number of evaluations.
  """
  C = TreeBasedTensor.randn(tree, shape, args.Ny, seed=args.rng_seed)
  cross = TreeCross(C, seed=args.rng_seed)
  cross.run(cfun, n_iter=args.cross_niter, eps=args.cross_eps, kickrank=args.cross_kickrank, verbose=args.verbose > 1)
  return C.round(args.cross_eps), cross.n_eval

def uses_cross(args: ARGS_tree_tensor_training) -> bool:
  """ Whether the initialization needs the cross approximation of the coefficient. """
  if args.rank_adapt: return False
  return args.init in ('cross', 'cross_tensor') or (args.init == 'mean' and args.init_ranks is None)

def split_data(args: ARGS_tree_tensor_training, values, shape, rng, device):
  """
  Training and validation data (indices and values of each) and the number of
  samples drawn: N_train + N_test, or with val_fraction one budget of N_train
  samples, of which the last max(2, round(val_fraction * N_train)) validate.
  """
  if args.val_fraction <= 0:
    train_ind, train_vals = sample_data(values, shape, args.N_train, rng, device)
    val_ind, val_vals = sample_data(values, shape, args.N_test, rng, device)
    return train_ind, train_vals, val_ind, val_vals, args.N_train + args.N_test
  N_val = max(2, round(args.val_fraction * args.N_train))
  if N_val >= args.N_train:
    raise ValueError(f'No training samples left: N_train = {args.N_train}, N_val = {N_val}')
  ind, vals = sample_data(values, shape, args.N_train, rng, device)
  N_fit = args.N_train - N_val
  return ind[:N_fit], vals[:N_fit], ind[N_fit:], vals[N_fit:], args.N_train

def reference_set(args: ARGS_tree_tensor_training, values, shape):
  """
  Indices and values of the reference set of the final MC errors: those of
  ref_file, N_mc fresh samples of the solution (target 'diffusion'), or None,
  None for the coefficient (its MC errors come from coeff_mc_errors).
  """
  if args.ref_file is not None:
    ref = torch.load(args.ref_file)
    ref_ind, ref_vals = ref['ind'], ref['vals']
    if tuple(ref_vals.shape[1:]) != shape[:1] or ref_ind.shape[1] != len(shape):
      raise ValueError(f'Reference set {args.ref_file} does not match the shape {shape}')
    return ref_ind, ref_vals
  if args.target != 'coeff':
    return sample_data(values, shape, args.N_mc, np.random.default_rng(seed=args.rng_seed + 1), 'cpu')
  return None, None

def initial_tensor(args: ARGS_tree_tensor_training, tree: Tree, shape, ranks, mean: float,
                   C: TreeBasedTensor = None, device=None) -> TreeBasedTensor:
  """
  Starting tensor of the training (see ARGS init and rank_adapt): ranks are the
  ranks by node id (or one rank), mean the mean of the training data, C the
  cross approximation (init 'cross_tensor').
  """
  if args.rank_adapt: # T-rank (1, ..., 1): the mean
    tensor = mean * TreeBasedTensor.ones(tree, shape)
  elif args.init == 'cross_tensor':
    return perturb(C, args.init_noise, seed=args.rng_seed, device=device)
  elif args.init == 'mean':
    # rank 1 constant plus random part, the ranks are ranks + 1 (until the first rounding, if any)
    tensor = mean * TreeBasedTensor.ones(tree, shape) \
      + TreeBasedTensor.randn(tree, shape, ranks, seed=args.rng_seed, scale=args.init_noise * abs(mean))
  else: # 'uniform', 'cross'
    return TreeBasedTensor.randn(tree, shape, ranks, seed=args.rng_seed, device=device)
  return TreeBasedTensor(tensor, device=device)

def perturb(tensor: TreeBasedTensor, noise: float, seed=None, device=None) -> TreeBasedTensor:
  """
  Copy of tensor with Gaussian noise added to every core, with standard deviation
  noise times the RMS of the core entries.
  """
  rng = torch.Generator()
  if seed is not None: rng.manual_seed(seed)
  cores = []
  for core in tensor.cores:
    scale = noise * torch.sqrt(torch.mean(core**2))
    cores += [core + scale * torch.randn(core.shape, generator=rng, dtype=core.dtype)]
  return TreeBasedTensor(cores, tensor.tree, device=device)

def enrich(tensor: TreeBasedTensor, node_ids, noise: float, seed=None) -> TreeBasedTensor:
  """
  Copy of tensor with the rank of the edge above every node in node_ids
  increased by one: the new column of the node's core and the new slice of its
  parent's core are Gaussian with standard deviation noise times the RMS of the
  core entries, so the tensor changes by O(noise^2).

  The copy is orthogonalized first (the norm in the root core). In an arbitrary
  gauge the directions of an edge can differ in norm by many orders of
  magnitude (trained tensors drift), and a random slice of the parent core then
  multiplies a huge direction with an O(1) weight.
  """
  rng = torch.Generator()
  if seed is not None: rng.manual_seed(seed)
  cores = list(TreeBasedTensor(tensor).orthogonalize().cores)
  def pad(core, axis):
    shape = list(core.shape)
    shape[axis] = 1
    scale = noise * torch.sqrt(torch.mean(core**2))
    return torch.cat([core, scale * torch.randn(shape, generator=rng, dtype=core.dtype).to(core.device)], dim=axis)
  for node_id in node_ids:
    node = tensor.tree.node_list[node_id]
    cores[node.id] = pad(cores[node.id], -1)
    cores[node.parent.id] = pad(cores[node.parent.id], node.child_ind)
  return TreeBasedTensor(cores, tensor.tree)

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

def local_gram_trace(model: ExplicitGradTreeTensor, node: TreeNode, E):
  """
  tr(A^T A) of the local linear map A: core -> f of node, i.e. the sum over the samples and the physical index
  of the product of the squared norms of the operands contracted with the core, times the sizes of the
  core modes no operand contracts with (the physical mode of the physical leaf).
  """
  ops = [(t * t, labels) for t, labels in model._env_ops(node, E)]
  tr = _contract(*ops, out=[])
  core_t, core_labels = model._core_op(node)
  env_labels = {l for _, labels in ops for l in labels}
  for size, label in zip(core_t.shape, core_labels):
    if label not in env_labels: tr = tr * size
  return tr

def lstsq_core(model: ExplicitGradTreeTensor, node: TreeNode, E, target, n_iter: int, precondition: bool = False,
               ridge: float = 0.):
  """
  Approximately solve the least squares problem min ||A(core) - target||^2 + mu ||core||^2 for the core of node
  (all other cores fixed) with n_iter iterations of CGLS, i.e. CG on the normal equations
  (A^T A + mu I) core = A^T target, starting from the current core. A and A^T are only applied through
  local contractions, A^T A is never formed. Steps are 0 once the residual of the normal
  equations decreased by CG_RTOL, so no values are moved to the host.

  The ridge parameter is relative to the mean eigenvalue of A^T A: mu = ridge * tr(A^T A) / numel(core),
  so it does not depend on the scale of the core (the gauge of the tree tensor).

  With precondition, CGLS runs on A S in z with core = S z, S = L^-T for a factorization L L^T of
  (an approximation of) A^T A + mu I: the exact block diagonal one for parameter leaves (leaf_gram_blocks),
  the Kronecker approximation otherwise (kron_gram_factors), which is exact for the physical leaf. For the
  Kronecker approximation every factor C_k is shifted by ridge^(1/K) tr(C_k) / r_k, so the product contains
  the relative shift ridge (plus positive semidefinite cross terms).
  """
  core = model.tensor.cores[node]
  mu = ridge * local_gram_trace(model, node, E) / core.numel() if ridge > 0 else 0.
  if not precondition:
    S = St = lambda x: x
  elif node.isleaf and node.dim != model.phys_dim:
    blocks = leaf_gram_blocks(model, node, E)
    if ridge > 0: blocks = blocks + mu * torch.eye(blocks.shape[-1], dtype=blocks.dtype, device=blocks.device)
    Linv = _inv_cholesky(blocks)
    S = lambda z: torch.einsum('ikj,ik->ij', Linv, z) # row i: L_i^-T z_i
    St = lambda g: torch.einsum('ijk,ik->ij', Linv, g)
  else:
    factors = kron_gram_factors(model, node, E)
    jitter = PRECOND_JITTER
    if ridge > 0: jitter = max(jitter, ridge**(1 / sum(C is not None for C in factors)))
    Linv = [None if C is None else _inv_cholesky(C, jitter) for C in factors]
    S = lambda z: _apply_modewise(z, Linv, transpose=True)
    St = lambda g: _apply_modewise(g, Linv)
  if ridge > 0:
    normal_res = lambda R: model.local_grad(node, R, E) - mu * core # A^T R - mu core
  else:
    normal_res = lambda R: model.local_grad(node, R, E)
  R = target - model.local_apply(node, core, E) # residual
  G = St(normal_res(R)) # residual of the (preconditioned) normal equations
  P = G
  gamma = torch.sum(G * G)
  gamma_stop = CG_RTOL**2 * gamma
  for _ in range(n_iter):
    active = gamma > gamma_stop
    SP = S(P)
    Q = model.local_apply(node, SP, E)
    QQ = torch.sum(Q * Q)
    if ridge > 0: QQ = QQ + mu * torch.sum(SP * SP)
    alpha = torch.where(active & (QQ > 0), gamma / QQ, 0.)
    core += alpha * SP
    R = R - alpha * Q
    G = St(normal_res(R))
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
    lstsq_core(model, node, E, target, args.epoch_inner, precondition=args.lstsq_precondition, ridge=args.ridge)
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

  @tensor.setter
  def tensor(self, tensor: TreeBasedTensor):
    """ Replaces the tensor (not copied). """
    self.model.tensor = tensor

  def sweep(self, ind, target, lr: float, args: ARGS_tree_tensor_training = None):
    """ One sweep over all cores, with args instead of the trainer's settings if given. """
    args = self.args if args is None else args
    with torch.no_grad():
      if args.orthogonalize: self.tensor.orthogonalize()
      self.model.set_indices(ind)
      leaf_core = self.tensor.cores[self.model.tree.dim2leaf(self.model.phys_dim)]
      # without reduction if it would not reduce the physical dimension (and change the rank of the leaf)
      if args.phys_reduction and leaf_core.shape[1] < leaf_core.shape[0]:
        reduced_sweep(self.model, target, args, lr)
      else:
        self.model.up_sweep()
        sweep(self.model, target, args, lr)

  def forward(self, ind):
    with torch.no_grad():
      return self.model.forward(ind)


def admissible(ranks, tree: Tree, shape) -> bool:
  """
  Necessary conditions for tree-based ranks (by node id, the root's is 1): a
  leaf rank is at most its mode size, and at every interior node each of its
  rank and its children's ranks is at most the product of the others.
  """
  rank = lambda node: 1 if node.isroot else int(ranks[node.id])
  for node in tree.node_list:
    if node.isleaf:
      if rank(node) > shape[node.dim]: return False
      continue
    group = [rank(node)] + [rank(c) for c in node.children]
    total = np.prod(group)
    if any(g * g > total for g in group): return False
  return True

def select_nodes(sigma, ranks, tree: Tree, shape, theta_star: float = .8, eps: float = 1e-12):
  """
  Nodes whose rank is increased (Grelier, Nouy, Chevreuil 2019, (22), (23)).

  Parameters
  ----------
  sigma: NodeIndexedList
    Singular values of the enriched tensor (ranks + 1) at every edge.
  ranks: sequence of int
    Current ranks by node id.
  tree: Tree
  shape: tuple of int
  theta_star: float, optional
    Upper bound of theta.
  eps: float, optional
    Nodes with eta <= eps ||u|| are no candidates.

  Returns
  -------
  selected: list of int
    Ids of the nodes with eta >= theta max(eta), theta the largest value
    <= theta_star such that the new ranks are admissible (empty if none).
  new_ranks: list of int
  eta: dict
    eta of every candidate node by id: its singular value number ranks + 1.
  """
  norm = next(float(torch.linalg.norm(s)) for s in sigma if s is not None)
  eta = {}
  for node in tree.node_list:
    if node.isroot: continue
    s, r = sigma[node], int(ranks[node.id])
    if len(s) <= r or (node.isleaf and r >= shape[node.dim]): continue
    if float(s[r]) > eps * norm: eta[node.id] = float(s[r])
  if not eta:
    return [], list(ranks), eta
  eta_max = max(eta.values())
  thetas = [theta_star] + sorted({e / eta_max for e in eta.values() if e / eta_max < theta_star}, reverse=True)
  for theta in thetas:
    selected = sorted(i for i, e in eta.items() if e >= theta * eta_max)
    new_ranks = [int(r) + (i in selected) for i, r in enumerate(ranks)]
    if admissible(new_ranks, tree, shape):
      return selected, new_ranks, eta
  return [], list(ranks), eta


# round_mode 'val': the largest tolerance within this relative margin of the best validation loss is chosen
VAL_ROUND_MARGIN = 1e-2

def val_round_tols(eps: float):
  """ Candidate tolerances of round_mode 'val': 0 (no rounding) and eps * 10^(k/2) <= 1. """
  n = int(np.floor(2 * np.log10(1 / eps) + 1e-9)) + 1 if eps < 1 else 1
  return [0.] + [eps * 10**(k / 2) for k in range(n)]

def mse_at(tensor: TreeBasedTensor, ind, vals) -> float:
  with torch.no_grad():
    return torch.nn.functional.mse_loss(ExplicitGradTreeTensor(tensor).forward(ind), vals).item()

def select_round_tol(tensor: TreeBasedTensor, ind, vals, eps: float):
  """
  Rounding tolerance in val_round_tols(eps) with the best validation loss (of the
  rounded tensor at ind, vals), the largest one within VAL_ROUND_MARGIN of the
  best. Returns it and the candidates [tol, loss, size].
  """
  candidates = []
  for tol in val_round_tols(eps):
    x = tensor if tol == 0 else TreeBasedTensor(tensor).round(tol)
    candidates += [[tol, mse_at(x, ind, vals), int(x.sparse_size)]]
  best = min(loss for _, loss, _ in candidates)
  tol = max(tol for tol, loss, _ in candidates if loss <= best * (1 + VAL_ROUND_MARGIN))
  return tol, candidates


def train(args: ARGS_tree_tensor_training, return_tensor: bool = False):
  """
  Fit a tree tensor to samples of the target with core-wise sweeps, optionally
  with rounding or rank adaptation.

  Returns the stats: per epoch losses, sizes and timings of the phases sweep,
  round and eval (t_elapsed excludes eval), and after training the Monte Carlo
  errors (RMS over the physical dimension per parameter sample, MC_rel_rms
  relative to the RMS of the target). The MC errors are at the samples of
  ref_file if given, otherwise at N_mc fresh samples.
  Train and test (= validation) loss are the MSE over the full data sets after
  the epoch (with rounding 'fixed': after rounding to extra_eps * eps), the
  final ones (final_train_loss, final_test_loss) after the final rounding. With
  round_mode 'val' the epochs of both phases are in the per epoch stats, the
  fine-tuning starts at epoch round_epoch. With rank_adapt the epochs of all
  fits are in the per epoch stats, the iterations in stats['adapt'].
  """
  if args.verbose: print('Running tree tensor training...')
  if args.verbose: print(args)
  if args.init not in INITS:
    raise ValueError(f'Unknown init {args.init}')
  if args.rounding and args.round_mode not in ROUND_MODES:
    raise ValueError(f'Unknown round_mode {args.round_mode}')
  if args.rank_adapt and args.rounding:
    raise ValueError('rank_adapt and rounding cannot be combined')
  if args.adapt_init not in ADAPT_INITS:
    raise ValueError(f'Unknown adapt_init {args.adapt_init}')
  if args.target != 'coeff' and args.init in ('cross', 'cross_tensor') and not args.rank_adapt:
    raise ValueError(f'init {args.init} uses the coefficient approximation, not possible for target {args.target}')

  device = torch.device(args.device)
  stats = {key: [] for key in ['train_loss', 'test_loss', 'size', 'max_rank', 'lr',
                               't_sweep', 't_round', 't_eval', 't_elapsed']}
  t = time.perf_counter()
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  cfun = lambda x : torch.from_numpy(cfun_np(x))
  values = target_values(args)
  shape = tensor_shape(args)
  tree = build_tree(args)
  stats.update(tree_indices(tree))

  t_data = time.perf_counter()
  rng = np.random.default_rng(seed=args.rng_seed)
  train_ind, train_eval, test_ind, test_eval, stats['n_samples'] = split_data(args, values, shape, rng, device)
  stats['N_fit'], stats['N_val'] = len(train_ind), len(test_ind)
  stats['t_data'] = time.perf_counter() - t_data

  # MC errors of the cross and the trained tensor (target 'coeff' without ref_file) at the same parameter samples
  mc_errors = lambda tensor: coeff_mc_errors(tensor, cfun, np.random.default_rng(seed=args.rng_seed + 1), args.N_mc)
  ref_ind, ref_vals = reference_set(args, values, shape)
  final_errors = lambda tensor: mc_errors(tensor) if ref_ind is None else ref_errors(tensor, ref_ind, ref_vals)
  # RMS of the target, estimated from the training data without a reference set
  stats['ref_rms'] = torch.sqrt(torch.mean((train_eval if ref_vals is None else ref_vals)**2)).item()
  ref_rel_err = lambda tensor: RMS(final_errors(tensor)) / stats['ref_rms']

  C, ranks = None, args.rinit if args.init_ranks is None else list(args.init_ranks)
  if uses_cross(args):
    t_cross = time.perf_counter()
    C, stats['cross_neval'] = cross_approximation(cfun, tree, coeff_shape(args), args)
    stats['t_cross'] = time.perf_counter() - t_cross
    ranks = ranks_of(C)
    stats['cross_size'], stats['cross_ranks'] = C.sparse_size, ranks
    errs = mc_errors(C)
    stats['cross_MC_rms'], stats['cross_MC_max'] = RMS(errs), np.max(errs)
    if args.verbose:
      print(f'Cross approximation: RMS l2 err = {stats["cross_MC_rms"]:.2e}, size {C.sparse_size}, '
            f'max rank {max(ranks)}, {stats["t_cross"]:.1f}s')
    # relative to the RMS of the coefficient, estimated from the training data (or the cross approximation)
    if args.target == 'coeff':
      coeff_rms = torch.sqrt(torch.mean(train_eval**2)).item()
    else:
      coeff_rms = (C.norm() / np.sqrt(C.size)).item()
    stats['cross_rel_err'] = stats['cross_MC_rms'] / coeff_rms
    if stats['cross_rel_err'] > CROSS_WARN_FACTOR * args.cross_eps:
      warnings.warn(f'Relative error {stats["cross_rel_err"]:.1e} of the cross approximation exceeds '
                    f'{CROSS_WARN_FACTOR} x cross_eps = {CROSS_WARN_FACTOR * args.cross_eps:.1e}, '
                    f'consider more sweeps (cross_niter = {args.cross_niter}).')

  tensor = initial_tensor(args, tree, shape, ranks, train_eval.mean().item(), C, device)
  del C
  stats['init_size'] = tensor.sparse_size
  stats['init_ranks'] = ranks_of(tensor)
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

  def evaluate():
    with torch.no_grad():
      train_loss = torch.nn.functional.mse_loss(trainer.forward(train_ind), train_eval).item()
      test_loss = torch.nn.functional.mse_loss(trainer.forward(test_ind), test_eval).item()
    return train_loss, test_loss

  t_elapsed = 0.
  stats['best_epochs'] = [] # keep_best: per phase the epoch of the returned tensor, None for the starting tensor
  stats['start_test_loss'] = [] # keep_best: per phase the validation loss of the starting tensor
  if args.track_ref: stats['ref_rel_err'], stats['start_ref_rel_err'] = [], []

  def train_phase(round_tol: float = None, epochs: int = None, keep_best: bool = None, ridge: float = None):
    """
    Sweeps until the lr schedule stops (at most epochs, default epoch_outer),
    rounding to round_tol after every sweep if given. With keep_best (default
    args.keep_best) the tensor with the best validation loss, the starting
    tensor included, is restored at the end. ridge overrides args.ridge.
    """
    nonlocal t_elapsed
    sweep_args = args if ridge is None else replace(args, ridge=ridge)
    keep_best = args.keep_best if keep_best is None else keep_best
    schedule = LRSchedule(args.lr_start, args.lr_factor, args.lr_patience, args.lr_threshold)
    lr = schedule.lr
    if args.track_ref: stats['start_ref_rel_err'] += [ref_rel_err(trainer.tensor)]
    if keep_best:
      _, best_loss = evaluate()
      best_tensor, best_epoch = TreeBasedTensor(trainer.tensor), None
      stats['start_test_loss'] += [best_loss]
    for epoch in range(args.epoch_outer if epochs is None else epochs):
      _, t_sweep = timed(lambda: trainer.sweep(train_ind, train_eval, lr, sweep_args))
      t_round = 0.
      if round_tol is not None and epoch >= args.round_start:
        _, t_round = timed(lambda: trainer.tensor.round(round_tol))
      (train_loss, test_loss), t_eval = timed(evaluate)
      t_elapsed += t_sweep + t_round

      record = dict(train_loss=train_loss, test_loss=test_loss, size=trainer.tensor.sparse_size,
                    max_rank=max(ranks_of(trainer.tensor)), lr=lr, t_sweep=t_sweep, t_round=t_round,
                    t_eval=t_eval, t_elapsed=t_elapsed)
      if args.track_ref: record['ref_rel_err'] = ref_rel_err(trainer.tensor)
      for key, value in record.items(): stats[key] += [value]
      if args.verbose:
        print('\r' + f'epoch {len(stats["train_loss"]) - 1}, train loss {train_loss:.2e}, '
              f'validation loss {test_loss:.2e}, lr {lr:.1e}, size {trainer.tensor.sparse_size}, '
              f'sweep {t_sweep:.2f}s', flush=True)

      if keep_best and test_loss < best_loss:
        best_tensor, t_copy = timed(lambda: TreeBasedTensor(trainer.tensor))
        best_loss, best_epoch = test_loss, len(stats['test_loss']) - 1
        t_elapsed += t_copy

      lr = schedule.step(test_loss)
      if lr < args.lr_end:
        break

    if keep_best:
      stats['best_epochs'] += [best_epoch]
      if best_epoch != len(stats['test_loss']) - 1: # otherwise the current tensor is the best one
        trainer.tensor = best_tensor

  def rank_one_correction(m):
    """
    Small random rank-1 term (init_noise times the RMS of the residual), the fit
    of u + w grows it from the data. A rank-1 least squares fit to the residual
    (as in the paper, with smooth leaf bases) extrapolates badly with parameter
    index leaves: it made the validation loss of u + w 20x worse.
    """
    with torch.no_grad():
      res = train_eval - trainer.forward(train_ind)
      return TreeBasedTensor.randn(tree, shape, 1, seed=args.rng_seed + m, device=device,
                                   scale=args.init_noise * torch.sqrt(torch.mean(res**2)).item())

  def rank_adaptation():
    """
    Grow the ranks from rank 1 (Grelier, Nouy, Chevreuil 2019, Algorithms 5 and
    6): per iteration fit u~ = u + rank-1 correction (ranks + 1, ridge
    adapt_select_ridge), increase the ranks of the nodes selected from the
    singular values of u~ (select_nodes) and fit u at the new ranks, starting
    from u with the selected edges padded by small random entries (adapt_init
    'enrich', u~ only selects the nodes) or from u~ truncated to the new ranks
    ('truncate', which inherits the overfitting of u~). Stops when no node is
    selected, after adapt_patience iterations without improvement or when the
    validation loss exceeds adapt_overfit times the best one. Restores the
    iterate with the best validation loss.
    """
    nonlocal t_elapsed
    stats['adapt'] = []
    def record(m, selected, eta):
      train_loss, val_loss = evaluate()
      entry = {'m': m, 'ranks': ranks_of(trainer.tensor), 'size': trainer.tensor.sparse_size,
               'train_loss': train_loss, 'val_loss': val_loss, 'selected': selected, 'eta': eta,
               'epoch': len(stats['train_loss']), 't_elapsed': t_elapsed}
      if args.track_ref: entry['ref_rel_err'] = ref_rel_err(trainer.tensor)
      stats['adapt'] += [entry]
      if args.verbose:
        print(f'rank adaptation {m}: max rank {max(entry["ranks"])}, size {entry["size"]}, '
              f'validation loss {val_loss:.2e}, {len(selected)} nodes increased', flush=True)
      return val_loss

    train_phase(epochs=args.adapt_epochs)
    best_loss, best_tensor, best_m = record(0, [], {}), TreeBasedTensor(trainer.tensor), 0
    for m in range(1, args.adapt_iter + 1):
      u = trainer.tensor
      w, t_w = timed(lambda: rank_one_correction(m))
      t_elapsed += t_w
      trainer.tensor = u + w
      # last epoch, not the best validation one: u~ only selects the ranks (with keep_best it often stays at
      # u + w, whose new singular values then come from the random w)
      train_phase(epochs=args.adapt_epochs, keep_best=False, ridge=args.adapt_select_ridge)
      sigma, t_s = timed(trainer.tensor.singular_values)
      selected, new_ranks, eta = select_nodes(sigma, ranks_of(u), tree, shape, args.adapt_theta)
      if not selected:
        trainer.tensor = u
        break
      # start of the next fit
      if args.adapt_init == 'enrich':
        u_next, t_r = timed(lambda: enrich(u, selected, args.init_noise, seed=args.rng_seed + m))
        trainer.tensor = u_next
      else: # u~ truncated
        _, t_r = timed(lambda: trainer.tensor.round(ranks=new_ranks))
      t_elapsed += t_s + t_r
      train_phase(epochs=args.adapt_epochs)
      val_loss = record(m, selected, eta)
      if val_loss < best_loss:
        best_loss, best_tensor, best_m = val_loss, TreeBasedTensor(trainer.tensor), m
      elif val_loss > args.adapt_overfit * best_loss or m - best_m >= args.adapt_patience:
        break
    stats['adapt_best'] = best_m
    trainer.tensor = best_tensor

  if args.rank_adapt:
    rank_adaptation()
  elif not args.rounding:
    train_phase()
  elif args.round_mode == 'fixed':
    # round during training to extra_eps * eps, finally to the target tolerance
    train_phase(args.extra_eps * args.eps)
    _, t_round = timed(lambda: trainer.tensor.round(args.eps))
    t_elapsed += t_round
  else: # 'val': train at full ranks, round to the validation selected tolerance, train again
    train_phase()
    (tol, candidates), t_round = timed(lambda: select_round_tol(trainer.tensor, test_ind, test_eval, args.eps))
    stats['val_round_tol'] = tol
    stats['val_round_candidates'] = candidates
    stats['round_epoch'] = len(stats['train_loss'])
    if tol > 0: trainer.tensor.round(tol)
    t_elapsed += t_round
    if args.verbose:
      print(f'Rounded to tol {tol:.1e} (validation selected), size {candidates[0][2]} -> {trainer.tensor.sparse_size}')
    train_phase()

  stats['n_epochs'] = len(stats['train_loss'])
  stats['final_train_loss'], stats['final_test_loss'] = evaluate()
  stats['t_train'] = t_elapsed
  stats['final_size'] = trainer.tensor.sparse_size
  stats['final_ranks'] = ranks_of(trainer.tensor)
  if device.type == 'cuda': stats['cuda_peak_MB'] = torch.cuda.max_memory_allocated(device) / 1024**2

  # tree tensor error estimate
  errs = final_errors(trainer.tensor)
  stats['MC_errors'] = errs
  stats['MC_rms'] = RMS(errs)
  stats['MC_max'] = np.max(errs)
  stats['MC_rel_rms'] = stats['MC_rms'] / stats['ref_rms']
  if args.verbose:
    print(f'Tree tensor RMS l2 err = {stats["MC_rms"]:.2e} (relative {stats["MC_rel_rms"]:.2e}), '
          f'training time {t_elapsed:.2f}s')

  if return_tensor:
    return stats, trainer.tensor
  return stats
