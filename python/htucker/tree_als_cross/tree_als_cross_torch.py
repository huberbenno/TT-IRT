import numpy as np
import torch
import copy
from scipy.sparse import issparse

from util.maxvol_torch import rect_maxvol
from tree_tensor.tree_tensor_torch import TreeBasedTensor, svd_cut_torch
from tree.tree import NodeIndexedList, TreeNode

class TreeALSCross:
  """
  ALS-cross solver for parametric linear systems A(y) u(y) = b(y) in tree-based
  tensor format.

  The matrix and the RHS depend affinely on coefficients, which are given as tree
  tensors over (coefficient index, y_1, ..., y_d). The solution u is a tree tensor over
  (spatial index, y_1, ..., y_d). Its spatial core is computed by solving the full
  system at a few parameter indices (via assem_solve_fun), all other cores by local
  systems, which are projected onto the solution basis towards the spatial core
  (Galerkin) and interpolated at maxvol indices away from it (cross).

  Dimension 0 (spatial) must be a leaf and the first child of the root. All tensors
  must share the tree, the parameter shape and the dtype (real or complex).

  Parameters
  ----------
  A_params: list[TreeBasedTensor]
    Coefficients of the matrix components, A_params[k] has shape (n_k, n_1, ..., n_d).
  b_params: list[TreeBasedTensor]
    Coefficients of the RHS components, b_params[k] has shape (m_k, n_1, ..., n_d).
  assem_solve_fun:
    Object implementing the spatial problem with Nx degrees of freedom:

    - matrix(cores): cores[k] is an (n_k, r_k) tensor. Returns a list with a list of
      r_k (Nx, Nx) matrices for each component k, assembled from the columns of cores[k].
    - rhs(cores): cores[k] is an (m_k, r_k) tensor. Returns a list with a list of r_k
      (Nx, 1) vectors for each component k.
    - solve(coeff_A, coeff_b): coeff_A[k] is (n_k, I) and coeff_b[k] is (m_k, I).
      Returns a list of the I (Nx, 1) solutions for the columns of the coefficients.
  use_indices: bool, optional
    Not used currently.
  kickrank: int, optional
    Rank of the residual based (AMEn) enrichment of the solution. 0 disables rank
    adaption, then the ranks can only decrease.
  rinit: int, optional
    Initial rank of the random solution. 0 uses the ranks of A_params[0].
  verbose: int, optional
    0 is silent, 1 prints a line per sweep, 2 additionally prints dx for every node.
  seed: int, optional
    Seed for the random initial solution and the random initial residual data.
    Default uses system entropy.

  Attributes
  ----------
  u: TreeBasedTensor
    Current solution. Internally the root is treated as a child of the spatial leaf
    (root_node), so its core has the spatial axis last. Use get_tensor for a regular
    tensor.
  n_eval: int
    Number of solves of the full spatial problem.
  iter_total: int
    Number of sweeps done.
  history: list[dict]
    Per sweep: sweep (index), n_eval (total solves after it), max_dx (maximal relative
    core change) and change (relative change of the solution, see run, None if not computed).
  stopped: bool
    Whether the last run stopped by its stopping criterion.
  """

  def __init__(
      self,
      A_params: list[TreeBasedTensor],
      b_params: list[TreeBasedTensor],
      assem_solve_fun,
      use_indices=False,
      kickrank=0,
      rinit=0,
      verbose=0,
      seed=None
    ):
    # store parameters
    self.A_params = [copy.deepcopy(A_param) for A_param in A_params]
    self.b_params = [copy.deepcopy(b_param) for b_param in b_params]
    self.assem_solve_fun = assem_solve_fun

    self.verbose = verbose
    self.kickrank = kickrank
    self.n_eval = 0
    self.iter_total = 0
    self.history = [] # one dict per sweep: sweep, n_eval, max_dx, change (None if not computed)
    self.stopped = False
    self._prev = None # (solution, squared norm) after the last sweep with tracked change

    self.dtype = A_params[0].dtype
    assert all(A_param.dtype == self.dtype for A_param in A_params), 'Incompatible data types.'
    assert all(b_param.dtype == self.dtype for b_param in b_params), 'Incompatible data types.'

    self.rng_numpy = np.random.default_rng(seed)
    self.rng_torch = torch.Generator()
    if seed is None:
      self.rng_torch.seed()
    else:
      self.rng_torch.manual_seed(seed)

    # tree and shape must be the same for all parameters (except spatial dim)
    # currently not checked
    self.tree = copy.deepcopy(A_params[0].tree)
    self.tree_mod = copy.deepcopy(A_params[0].tree)
    self.shape = A_params[0].shape[1:]

    self.M_A = len(self.A_params)
    self.M_b = len(self.b_params)

    # interpolatory form towards dim 0, keep the maxvol indices of the first matrix parameter
    self.A_params[0], indexset_list, indexset_dims_list, maxvol_ind_list = self._orth_towards_0(self.A_params[0])
    for k in range(1, self.M_A):
      self.A_params[k] = self._orth_towards_0(self.A_params[k], return_indices=False)
    for k in range(self.M_b):
      self.b_params[k] = self._orth_towards_0(self.b_params[k], return_indices=False)

    # modified root for more generic code
    # special core must be leaf and first child of root
    self.root_node = self.tree_mod.root
    self.root_node.parent = self.root_node.children[0]
    self.root_node.children = tuple(self.root_node.children[1:])

    for param in self.A_params + self.b_params:
      self._to_modified_root(param)

    # init matrix and rhs variables
    A0_cores = [A_param.cores[A_param.tree.dim2id(0)] for A_param in self.A_params]
    b0_cores = [b_param.cores[b_param.tree.dim2id(0)] for b_param in self.b_params]
    self.A0 = assem_solve_fun.matrix(A0_cores)
    self.F0 = [torch.hstack(F0k) for F0k in assem_solve_fun.rhs(b0_cores)]
    self.Nx = self.A0[0][0].shape[1]

    if rinit > 0:
      self.u = TreeBasedTensor.randn(tree=self.tree, shape=(self.Nx,) + self.shape, rank=rinit,
                                     dtype=self.dtype, seed=int(self.rng_numpy.integers(2**63)))
      self.u, indexset_list, indexset_dims_list, maxvol_ind_list = self._orth_towards_0(self.u)
      self._to_modified_root(self.u)
    else:
      cores = NodeIndexedList([torch.randn(c.shape, generator=self.rng_torch, dtype=self.dtype) for c in self.A_params[0].cores])
      cores[self.tree.dim2leaf(0)] = torch.randn((self.Nx, cores[self.tree.dim2leaf(0)].shape[1]), generator=self.rng_torch, dtype=self.dtype)
      self.u = TreeBasedTensor(cores, self.tree)

    # init right proj UA and Ub (eval at maxvol)
    self.UA = [self._partial_evals(A_param, maxvol_ind_list) for A_param in self.A_params]
    self.Ub = [self._partial_evals(b_param, maxvol_ind_list) for b_param in self.b_params]

    # init UAU and UF
    self._init_right_projection()

    if self.kickrank > 0:
      self._init_AMEn_data()

    if self.verbose > 0:
      print('Init finished')

  def run(self, n_iter=1, tol=1e-3, stop_tol=None, track_change=False):
    """
    Run sweeps over the tree, starting at the spatial core. Can be called repeatedly
    to continue the iteration.

    Parameters
    ----------
    n_iter: int, optional
      Number of sweeps, the maximal number with stop_tol.
    tol: float, optional
      Relative truncation tolerance, split evenly over the cores
      (tol / sqrt(order) for each truncation).
    stop_tol: float, optional
      Stop after the first sweep in which the relative change of the solution
      ||u_k - u_{k-1}|| / ||u_k|| (Frobenius norm over all entries) is at most stop_tol,
      u_{k-1} being the solution before the sweep. The change is computed from inner
      products of the tree tensors, so it needs no solves. Default: no stopping criterion.
    track_change: bool, optional
      Compute the change after every sweep (into history) also without stop_tol.

    Sets stopped (True if the stopping criterion was met) and appends one entry per sweep
    to history.
    """
    self.tol = tol
    self.max_dx = 0
    self.stopped = False
    compute_change = stop_tol is not None or track_change
    special = self.tree.root.children[0]

    for iteration in range(self.iter_total, self.iter_total + n_iter):
      if compute_change and self._prev is None:
        self._prev = self._with_norm(self.get_tensor())
      if self.verbose > 0:
        print(f'= swp={iteration}')

      #### special core: solve at the parameter indices of root_node
      arg_A = [torch.tensordot(self.A_params[k].cores[special], self.UA[k][self.root_node], dims=((-1,),(-1,)))
               for k in range(self.M_A)]
      arg_b = [torch.tensordot(self.b_params[k].cores[special], self.Ub[k][self.root_node], dims=((-1,),(-1,)))
               for k in range(self.M_b)]
      U0 = torch.hstack(self.assem_solve_fun.solve(arg_A, arg_b))
      self.n_eval += U0.shape[1]
      U_prev = self.u.cores[special]
      self._report(special, torch.linalg.norm(U0 - U_prev) / torch.linalg.norm(U0), '(s)')

      # truncate, the non-orth factor goes to root_node
      U0, v = self._truncate(U0)

      ## rank adaption: residual at the residual indices of root_node
      if self.kickrank > 0:
        rz = self.ZA[0][self.root_node].shape[0]
        Z0 = torch.zeros((self.Nx, rz), dtype=self.dtype)
        cru = torch.tensordot(U0 @ v, self.ZU[self.root_node], dims=((-1,),(-1,)))
        for k in range(self.M_A):
          if issparse(self.A0[k][0]):
            raise NotImplementedError
          for l in range(rz):
            crA = torch.zeros((self.Nx, self.Nx), dtype=self.dtype)
            for j, A0_j in enumerate(self.A0[k]):
              crA += A0_j * self.ZA[k][self.root_node][l,j]

            Z0[:,l] += torch.ravel(crA @ cru[:,l])

        for k in range(self.M_b):
          Z0 -= torch.tensordot(self.F0[k], self.Zb[k][self.root_node], dims=((-1,),(-1,)))

        Z0 = torch.linalg.qr(Z0)[0]
        U0, v = self._enrich(U0, v, Z0)

      self.u.cores[special] = U0
      self._cast_into(self.root_node, v)

      ## projections onto the solution basis U0 (and the residual basis Z0)
      UAU, UF = self._project_spatial(U0, U0)
      for k in range(self.M_A): self.UAU[k][special] = UAU[k]
      for k in range(self.M_b): self.UF[k][special] = UF[k]
      if self.kickrank > 0:
        ZUA, ZUb = self._project_spatial(Z0, U0)
        for k in range(self.M_A): self.ZUA[k][special] = ZUA[k]
        for k in range(self.M_b): self.ZUb[k][special] = ZUb[k]

      ## traverse tree
      self._als_worker(self.root_node)

      change = self._change() if compute_change else None
      if not compute_change: self._prev = None # the next tracked sweep compares with the solution before it
      self.history.append({'sweep': iteration, 'n_eval': self.n_eval, 'max_dx': float(self.max_dx), 'change': change})
      if self.verbose > 0:
        print(f'= swp={iteration} finished, max_dx={self.max_dx:.2e}' + ('' if change is None else f', change={change:.2e}'))
      self.max_dx = 0
      self.iter_total += 1

      if stop_tol is not None and change <= stop_tol:
        self.stopped = True
        if self.verbose > 0: print(f'= stopped: change {change:.2e} <= {stop_tol:.2e}')
        break

  def _with_norm(self, u):
    return u, float(torch.real(u.inner(u)))

  def _change(self):
    """ Relative change ||u - u_prev|| / ||u|| of the solution since the last tracked sweep, which becomes u_prev. """
    (u, uu), (v, vv) = self._with_norm(self.get_tensor()), self._prev
    self._prev = (u, uu)
    if uu == 0: return float('inf')
    return float(np.sqrt(max(uu + vv - 2 * float(torch.real(u.inner(v))), 0.) / uu))

  def get_tensor(self):
    """
    Current solution as a standalone tensor.

    Returns
    -------
    tensor: TreeBasedTensor
      Copy of the solution with shape (Nx, n_1, ..., n_d) on the original tree.
    """
    tensor = TreeBasedTensor(self.u)
    root_core = tensor.cores[tensor.tree.root]
    root_core = torch.unsqueeze(torch.moveaxis(root_core, -1, 0), -1)
    tensor.cores[tensor.tree.root] = root_core
    return tensor

  #### sweep

  def _als_worker(self, node: TreeNode):
    """
    Recursive sweep through the subtree below a node.

    An interior node is solved and then, for each child, truncated (and enriched)
    towards the child before recursing into it. Finally every node is solved again and
    truncated towards its parent, see _step_up.

    Parameters
    ----------
    node: TreeNode
      Node of the modified tree (root_node or below).
    """
    if not node.isleaf:
      self._report(node, self._solve_reduced(node), 'down')
      for child_ind, child in enumerate(node.children):
        self._step_down(node, child_ind, child)
        self._als_worker(child)

    self._report(node, self._solve_reduced(node), 'leaf' if node.isleaf else 'up')
    self._step_up(node)

  def _step_down(self, node: TreeNode, child_ind: int, child: TreeNode):
    """
    Move the non-orthogonal part of the core of node into a child: truncate (and
    enrich) the core towards the child, then update the projections UAU, UF (and ZUA,
    ZUb) of everything except the child's subtree.
    """
    other_axes = [a for a in range(node.n_children + 1) if a != child_ind]
    core = torch.moveaxis(self.u.cores[node], child_ind, -1)
    shape = core.shape
    cru, v = self._truncate(core.reshape(-1, shape[-1]))

    ## AMEn rank adaption
    if self.kickrank > 0:
      U = torch.moveaxis((cru @ v).reshape(shape[:-1] + (-1,)), -1, child_ind)
      interp = self._interp_items(node, self.ZU, self.ZA, self.Zb, [child_ind])
      # residual projected onto the current basis (enrichment) and onto the residual basis (update)
      crz = self._residual(node, U, interp, self._proj_items(node, self.UAU, self.UF, other_axes))
      crz_new = self._residual(node, U, interp, self._proj_items(node, self.ZUA, self.ZUb, other_axes))
      crz = torch.moveaxis(crz, child_ind, -1)
      cru, v = self._enrich(cru, v, crz.reshape(-1, crz.shape[-1]))

    self._cast_into(child, v)
    self.u.cores[node] = torch.moveaxis(cru.reshape(shape[:-1] + (-1,)), -1, child_ind)

    ## projections onto the solution basis (and the residual basis) for the child
    u = self.u.cores[node]
    UAU, UF = self._project(node, torch.conj(u), u, child_ind, self._proj_items(node, self.UAU, self.UF, other_axes))
    for k in range(self.M_A): self.UAU[k][node] = UAU[k]
    for k in range(self.M_b): self.UF[k][node] = UF[k]

    if self.kickrank > 0:
      crz_new = torch.moveaxis(crz_new, child_ind, -1)
      zshape = crz_new.shape[:-1]
      crz_new = torch.linalg.qr(crz_new.reshape(-1, crz_new.shape[-1]))[0]
      crz_new = torch.moveaxis(crz_new.reshape(zshape + (-1,)), -1, child_ind)
      ZUA, ZUb = self._project(node, torch.conj(crz_new), u, child_ind, self._proj_items(node, self.ZUA, self.ZUb, other_axes))
      for k in range(self.M_A): self.ZUA[k][node] = ZUA[k]
      for k in range(self.M_b): self.ZUb[k][node] = ZUb[k]
      # TODO update Zb, ZA ?

  def _step_up(self, node: TreeNode):
    """
    Move the non-orthogonal part of the core of node into its parent: truncate (and
    enrich) the core towards the parent, select maxvol indices, and update the
    interpolation (UA, Ub) and projections (UAU, UF) of the subtree, as well as the
    residual data (ZUA, ZUb, ZU, ZA, Zb). Works the same for leaves and interior nodes.
    """
    shape = self.u.cores[node].shape
    parent_axis = len(shape) - 1
    child_axes = list(range(node.n_children))
    cru, v = self._truncate(self.u.cores[node].reshape(-1, shape[-1]))

    ## rank adaption
    if self.kickrank > 0:
      U = (cru @ v).reshape(shape[:-1] + (-1,))
      parent_proj = self._proj_items(node, self.ZUA, self.ZUb, [parent_axis])
      if node.isleaf:
        # the mode axis is not interpolated, current and update residual coincide
        I = torch.eye(shape[0], dtype=self.dtype)
        crz = self._residual(node, U, {0: (I, [I] * self.M_A, [I] * self.M_b)}, parent_proj)
        crz_new = crz
      else:
        # current: children at the solution indices, update: at the residual indices
        current = {a: (torch.eye(shape[a], dtype=self.dtype), [UA[c] for UA in self.UA], [Ub[c] for Ub in self.Ub])
                   for a, c in enumerate(node.children)}
        crz = self._residual(node, U, current, parent_proj)
        crz_new = self._residual(node, U, self._interp_items(node, self.ZU, self.ZA, self.Zb, child_axes), parent_proj)
      cru, v = self._enrich(cru, v, crz.reshape(-1, crz.shape[-1]))

    # maxvol
    ind, C = rect_maxvol(cru, maxK=cru.shape[1], warn_budget=False)
    C = C.to(dtype=self.dtype)
    qmax = cru[ind]
    self.u.cores[node] = C.reshape(shape[:-1] + (-1,))

    # cast non-orth factor to parent
    # TODO get rid of this hack, this stems from the tree modification
    ci = 1 if node.id == self.root_node.id else node.child_ind
    self._cast_into(node.parent, qmax @ v, ci)

    ## interpolation (sample params at the new indices) and projections of the subtree
    for k in range(self.M_A):
      self.UA[k][node] = self._eval_children(self.A_params[k].cores[node], [self.UA[k][c] for c in node.children])[ind]
    for k in range(self.M_b):
      self.Ub[k][node] = self._eval_children(self.b_params[k].cores[node], [self.Ub[k][c] for c in node.children])[ind]
    self._update_subtree_projections(node)

    # TODO update index set (for index based assem_solve_fun)

    ## residual basis and residual indices
    if self.kickrank > 0:
      zshape = crz_new.shape[:-1]
      crz_new = torch.linalg.qr(crz_new.reshape(-1, crz_new.shape[-1]))[0]
      ind, _ = rect_maxvol(crz_new, maxK=crz_new.shape[1], warn_budget=False)

      u = self.u.cores[node]
      crz_new_conj = torch.conj(crz_new).reshape(zshape + (-1,))
      ZUA, ZUb = self._project(node, crz_new_conj, u, parent_axis, self._proj_items(node, self.ZUA, self.ZUb, child_axes))
      for k in range(self.M_A): self.ZUA[k][node] = ZUA[k]
      for k in range(self.M_b): self.ZUb[k][node] = ZUb[k]
      self._sample_residual_indices(node, ind)

  def _solve_reduced(self, node:TreeNode):
    """
    Solve the local system for the core of a node and store it in u.

    The children are interpolated at their index sets (UA, Ub), the parent side is
    projected onto the solution basis (UAU, UF). This gives one small system per
    combination of child indices (per mode index for a leaf), solved as a batch.

    Parameters
    ----------
    node: TreeNode
      Node of the modified tree.

    Returns
    -------
    dx: torch.tensor
      Relative change of the core (0-dim).
    """
    parent = node.parent
    crA = [self._eval_children(self.A_params[k].cores[node], [self.UA[k][c] for c in node.children])
           for k in range(self.M_A)]
    crF = sum(self._eval_children(self.b_params[k].cores[node], [self.Ub[k][c] for c in node.children]) @ self.UF[k][parent].T
              for k in range(self.M_b))
    Ai = sum(torch.einsum('ija,na->nij', self.UAU[k][parent], crA[k]) for k in range(self.M_A))
    cru = torch.linalg.solve(Ai, crF)

    # check error
    dx = torch.linalg.norm(cru.flatten() - self.u.cores[node].flatten()) / torch.linalg.norm(cru)

    # update solution
    self.u.cores[node] = cru.reshape(self.u.cores[node].shape)

    return dx

  #### helpers

  def _report(self, node: TreeNode, dx, label: str):
    """ Track the maximal change of the sweep, print it for verbose > 1. """
    dx = float(dx)
    self.max_dx = max(self.max_dx, dx)
    if self.verbose > 1:
      print(f'    node {node.id} {label}'.ljust(20), f'dx={dx:.2e}')

  def _truncate(self, mat):
    """
    Truncated SVD of mat with tolerance tol / sqrt(order).

    Returns
    -------
    cru: torch.tensor
      Orthonormal left factor (n, r).
    v: torch.tensor
      Non-orthogonal right factor (r, m), mat ~ cru @ v.
    """
    cru, s, v = svd_cut_torch(mat, tol=self.tol/np.sqrt(self.tree.order))
    return cru, s[:, None] * v # broadcasting, since s is real also for complex v

  @staticmethod
  def _enrich(cru, v, crz):
    """
    Enrich the basis cru (n, r) by the columns of crz (n, rz) and reorthogonalize.
    Returns the new basis (n, r + rz) and the new right factor, such that
    new cru @ new v = cru @ v.
    """
    ru = cru.shape[-1]
    cru, rv = torch.linalg.qr(torch.cat((cru, crz), dim=-1))
    return cru, rv[:, :ru] @ v

  def _cast_into(self, node: TreeNode, M, axis=-1):
    """ Multiply the core of node along axis by the matrix M (r_new, r_old). """
    core = torch.tensordot(self.u.cores[node], M, dims=((axis,), (-1,)))
    self.u.cores[node] = torch.moveaxis(core, -1, axis)

  def _to_modified_root(self, tensor: TreeBasedTensor):
    """ Root core (r_0, r_1, ..., 1) -> (r_1, ..., r_0) for the modified tree (in place). """
    root = self.root_node
    tensor.cores[root] = torch.moveaxis(tensor.cores[root].squeeze(dim=-1), 0, -1)

  def _project_spatial(self, test, trial):
    """
    Projections of the spatial matrices and RHS: (conj(test).T A0_j trial)_j and
    conj(left).T F0, as lists over the components.
    """
    test_H = torch.conj(test.T)
    PA = [torch.stack([test_H @ A0_j @ trial for A0_j in A0k], dim=-1) for A0k in self.A0]
    Pb = [test_H @ F0k for F0k in self.F0]
    return PA, Pb

  @staticmethod
  def _eval_children(core, child_mats):
    """
    Contract the child axes of a core with matrices (rows_i, r_child_i).

    Returns
    -------
    mat: torch.tensor
      Matrix (prod_i rows_i, r_parent), first child slowest. For a leaf (no child
      matrices) the core itself.
    """
    for M in child_mats:
      core = torch.tensordot(core, M, dims=((0,), (-1,)))
    if child_mats:
      core = torch.moveaxis(core, 0, -1)
    return core.reshape(-1, core.shape[-1])

  @staticmethod
  def _neighbour(node: TreeNode, axis: int):
    """ Neighbour of node along a core axis: the child, or the parent for the last axis. """
    return node.children[axis] if axis < node.n_children else node.parent

  def _proj_items(self, node: TreeNode, PA, Pb, axes):
    """ {axis: ([PA[k][nb] for k], [Pb[k][nb] for k])} with the neighbours nb along the axes. """
    items = {}
    for a in axes:
      nb = self._neighbour(node, a)
      items[a] = ([P[nb] for P in PA], [P[nb] for P in Pb])
    return items

  def _interp_items(self, node: TreeNode, ZU, ZA, Zb, axes):
    """ {axis: (ZU[nb], [ZA[k][nb] for k], [Zb[k][nb] for k])} with the neighbours nb along the axes. """
    items = {}
    for a in axes:
      nb = self._neighbour(node, a)
      items[a] = (ZU[nb], [Z[nb] for Z in ZA], [Z[nb] for Z in Zb])
    return items

  def _project(self, node: TreeNode, test_conj, trial, out: int, proj):
    """
    Project the matrix and RHS parts of a node onto one axis.

    Parameters
    ----------
    node: TreeNode
      Node of the modified tree.
    test_conj, trial: torch.tensor
      Conjugated test and trial cores, same layout as the solution core of node.
    out: int
      Axis of the result.
    proj: dict
      {axis: (PA, Pb)} for all other axes, PA[k] (r_test, r_trial, rA_k) and
      Pb[k] (r_test, rb_k) projections of the neighbour along the axis (see _proj_items).
      Unused for a leaf, its mode axis is summed directly.

    Returns
    -------
    PA, Pb: list[torch.tensor]
      For each component, (r_test, r_trial, rA_k) and (r_test, rb_k) on axis out.
    """
    if node.isleaf:
      PA = [torch.einsum('ab,ac,ad->bcd', test_conj, trial, A.cores[node]) for A in self.A_params]
      Pb = [torch.tensordot(test_conj, b.cores[node], dims=((0,),(0,))) for b in self.b_params]
      return PA, Pb

    m = node.n_children + 1
    PA = []
    for k in range(self.M_A):
      args = [test_conj, np.arange(0, 3*m, 3), trial, np.arange(1, 3*m, 3), self.A_params[k].cores[node], np.arange(2, 3*m, 3)]
      for a, (P, _) in proj.items():
        args += [P[k], [3*a, 3*a+1, 3*a+2]]
      PA.append(torch.einsum(*args, [3*out, 3*out+1, 3*out+2]))
    Pb = []
    for k in range(self.M_b):
      args = [test_conj, np.arange(0, 2*m, 2), self.b_params[k].cores[node], np.arange(1, 2*m, 2)]
      for a, (_, P) in proj.items():
        args += [P[k], [2*a, 2*a+1]]
      Pb.append(torch.einsum(*args, [2*out, 2*out+1]))
    return PA, Pb

  def _residual(self, node: TreeNode, U, interp, proj):
    """
    Residual A u - b of a node core U, with the residual index on every axis.

    Parameters
    ----------
    node: TreeNode
      Node of the modified tree.
    U: torch.tensor
      Solution core, same layout as the core of node.
    interp: dict
      {axis: (ZU, ZA, Zb)} for axes evaluated at index sets: ZU (z, r_u) the solution,
      ZA[k] (z, rA_k) and Zb[k] (z, rb_k) the parameters at the indices.
    proj: dict
      {axis: (PA, Pb)} for the projected axes, PA[k] (z, r_u, rA_k) and Pb[k] (z, rb_k).

    Returns
    -------
    crz: torch.tensor
      Residual with the size z of interp or proj on every axis.
    """
    m = U.ndim
    crz = 0
    for k in range(self.M_A):
      args = [U, np.arange(1, 3*m, 3), self.A_params[k].cores[node], np.arange(2, 3*m, 3)]
      for a, (ZU, ZA, _) in interp.items():
        args += [ZU, [3*a, 3*a+1], ZA[k], [3*a, 3*a+2]]
      for a, (PA, _) in proj.items():
        args += [PA[k], [3*a, 3*a+1, 3*a+2]]
      crz = crz + torch.einsum(*args, np.arange(0, 3*m, 3))
    for k in range(self.M_b):
      args = [self.b_params[k].cores[node], np.arange(1, 2*m, 2)]
      for a, (_, _, Zb) in interp.items():
        args += [Zb[k], [2*a, 2*a+1]]
      for a, (_, Pb) in proj.items():
        args += [Pb[k], [2*a, 2*a+1]]
      crz = crz - torch.einsum(*args, np.arange(0, 2*m, 2))
    return crz

  def _update_subtree_projections(self, node: TreeNode):
    """
    Projections UAU[k][node] (conj(U), U, A_k) and UF[k][node] (conj(U), b_k) of the
    subtree below node onto the solution basis, from those of the children.
    The conjugated basis is always in the first slot.
    """
    u = self.u.cores[node]
    proj = self._proj_items(node, self.UAU, self.UF, range(node.n_children))
    UAU, UF = self._project(node, torch.conj(u), u, u.ndim - 1, proj)
    for k in range(self.M_A): self.UAU[k][node] = UAU[k]
    for k in range(self.M_b): self.UF[k][node] = UF[k]

  def _sample_residual_indices(self, node: TreeNode, ind):
    """ Solution and parameters of the subtree at the residual indices (ZU, ZA, Zb). """
    ch = node.children
    self.ZU[node] = self._eval_children(self.u.cores[node], [self.ZU[c] for c in ch])[ind]
    for k in range(self.M_A):
      self.ZA[k][node] = self._eval_children(self.A_params[k].cores[node], [self.ZA[k][c] for c in ch])[ind]
    for k in range(self.M_b):
      self.Zb[k][node] = self._eval_children(self.b_params[k].cores[node], [self.Zb[k][c] for c in ch])[ind]

  #### initialization

  def _init_right_projection(self):
    """
    Initialize the projections UAU[k][node] (conj(U), U, A_k) and UF[k][node]
    (conj(U), b_k) of all subtrees below root_node onto the initial solution basis.
    The conjugated basis is always in the first slot.
    """
    self.UAU = [NodeIndexedList(self.tree.n_nodes * [None]) for k in range(self.M_A)]
    self.UF = [NodeIndexedList(self.tree.n_nodes * [None]) for k in range(self.M_b)]

    def worker(node: TreeNode):
      for child in node.children:
        worker(child)
      self._update_subtree_projections(node)

    worker(self.root_node)

  def _partial_evals(self, tensor: TreeBasedTensor, maxvol_ind_list):
    """
    Evaluate the subtrees of a tensor at the index sets given by maxvol indices.

    Parameters
    ----------
    tensor: TreeBasedTensor
      Tensor with the root core in the modified layout.
    maxvol_ind_list: NodeIndexedList
      Maxvol indices of each node, into the rows of its leaf core or into the product
      of the children's index sets (first child slowest).

    Returns
    -------
    partial_evals: NodeIndexedList
      Matrix (n_ind, r) for each node below root_node, with the subtree evaluated at
      the n_ind indices.
    """
    partial_evals = NodeIndexedList(tensor.tree.n_nodes * [None])

    def worker(node):
      for child in node.children:
        worker(child)
      mat = self._eval_children(tensor.cores[node], [partial_evals[c] for c in node.children])
      partial_evals[node] = mat[maxvol_ind_list[node]]

    worker(self.root_node)
    return partial_evals

  def _init_AMEn_data(self):
    """
    Initialize the data for the residual enrichment: the solution and parameters at
    random residual indices (ZU, ZA, Zb) and random residual projections (ZUA, ZUb).
    """
    self.ZUA = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.A_params] # rz x ru x rA
    self.ZUb = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.b_params] # rz x rb
    self.ZU = NodeIndexedList(self.tree.n_nodes * [None]) # rz x ru
    self.ZA = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.A_params] # rz x rA
    self.Zb = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.b_params] # rz x rb

    def worker(node):
      for child in node.children:
        worker(child)

      ru = self.u.cores[node].shape[-1]
      if node.isleaf:
        n_rows, rz = self.u.cores[node].shape[0], self.kickrank
      else:
        n_rows, rz = int(np.prod([self.ZU[c].shape[0] for c in node.children])), min(ru, self.kickrank)
      # no solution yet, random indices and projections
      ind = self.rng_numpy.choice(np.arange(n_rows), rz, replace=True) #TODO replace=False?

      ch = node.children
      self.ZU[node] = self._eval_children(self.u.cores[node], [self.ZU[c] for c in ch])[ind]
      for k in range(self.M_A):
        rA = self.A_params[k].cores[node].shape[-1]
        self.ZUA[k][node] = torch.randn((rz, ru, rA), generator=self.rng_torch, dtype=self.dtype)
        self.ZA[k][node] = self._eval_children(self.A_params[k].cores[node], [self.ZA[k][c] for c in ch])[ind]
      for k in range(self.M_b):
        rb = self.b_params[k].cores[node].shape[-1]
        self.ZUb[k][node] = torch.randn((rz, rb), generator=self.rng_torch, dtype=self.dtype)
        self.Zb[k][node] = self._eval_children(self.b_params[k].cores[node], [self.Zb[k][c] for c in ch])[ind]

    worker(self.root_node)

  @staticmethod
  def _orth_towards_0(tensor: TreeBasedTensor, return_indices = True):
    """
    Bring a tensor into interpolatory form towards the leaf of dimension 0.

    All cores except the leaf of dimension 0 become interpolation matrices at maxvol
    indices, the leaf of dimension 0 holds the values. The tensor is modified in place.

    Parameters
    ----------
    tensor: TreeBasedTensor
      Tensor with dimension 0 in the first child of the root.
    return_indices: bool, optional
      Also return the index sets.

    Returns
    -------
    tensor: TreeBasedTensor
      The modified tensor.
    indexset_list: NodeIndexedList
      Multi-indices of the index set of each node (only if return_indices).
    indexset_dims_list: NodeIndexedList
      Dimensions of the multi-indices of each node (only if return_indices).
    maxvol_ind_list: NodeIndexedList
      Maxvol indices of each node, see _partial_evals (only if return_indices).
    """

    # orth towards root
    _, indexset_list, indexset_dims_list, maxvol_ind_list = tensor._orth_subtree_maxvol(tensor.tree.root)

    # assume dim 0 lives in first child of root
    # orth towards dim 0
    core = tensor.cores[tensor.tree.root]
    old_shape = core.shape
    core = core.reshape(core.shape[0], -1)
    q,r = torch.linalg.qr(core.T)
    ind, C = rect_maxvol(q, maxK=q.shape[-1], warn_budget=False)
    C = C.to(dtype=tensor.dtype)
    qmax = q[ind]
    tensor.cores[tensor.tree.root] = C.T.reshape((-1, ) + old_shape[1:])
    child0 = tensor.tree.root.children[0]
    tensor.cores[child0] = tensor.cores[child0] @ (qmax @ r).T

    if return_indices:
      # build indexset
      indexset, indexset_dims = np.zeros((1,0)), []
      for child in tensor.tree.root.children[1:]:
        indexset_dims = np.concatenate((indexset_dims, indexset_dims_list[child]))
        indexset = np.hstack(
          (np.repeat(indexset, repeats=indexset_list[child].shape[0], axis=0),
          np.tile(indexset_list[child], reps=(indexset.shape[0], 1)))
        )

      # put the new indexset into root spot
      # TODO reconsider
      indexset_list[tensor.tree.root] = indexset[ind]
      indexset_dims_list[tensor.tree.root] = indexset_dims
      maxvol_ind_list[tensor.tree.root] = ind

      return tensor, indexset_list, indexset_dims_list, maxvol_ind_list
    else:
      return tensor
