import numpy as np
import copy
from scipy.sparse import csr_matrix, issparse
from tt.maxvol import rect_maxvol

from tree_tensor.tree_tensor import TreeBasedTensor, svd_cut
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

    - matrix(cores): cores[k] is an (n_k, r_k) array. Returns a list with a list of
      r_k (Nx, Nx) matrices (dense or scipy.sparse) for each component k, assembled
      from the columns of cores[k].
    - rhs(cores): cores[k] is an (m_k, r_k) array. Returns a list with a list of r_k
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

    self.rng = np.random.default_rng(seed)

    # tree and shape must be the same for all parameters (except spatial dim)
    # currently not checked
    self.tree = copy.deepcopy(A_params[0].tree)
    self.tree_mod = copy.deepcopy(A_params[0].tree)
    self.shape = A_params[0].shape[1:]

    self.M_A = len(self.A_params)
    self.M_b = len(self.b_params)

    for k, A_param in enumerate(self.A_params):
      # keep maxvol indices of first param TT
      if k == 0:
        self.A_params[k], indexset_list, indexset_dims_list, maxvol_ind_list = self._orth_towards_0(A_param)
      else:
        self.A_params[k] = self._orth_towards_0(A_param, return_indices=False)

    for k, b_param in enumerate(self.b_params):
      self.b_params[k] = self._orth_towards_0(b_param, return_indices=False)

    # modified root for more generic code
    # special core must be leaf and first child of root
    self.root_node = self.tree_mod.root
    self.root_node.parent = self.root_node.children[0]
    self.root_node.children = tuple(self.root_node.children[1:])

    for k in range(self.M_A):
      self.A_params[k].cores[self.root_node] = np.moveaxis(self.A_params[k].cores[self.root_node].squeeze(axis=-1), 0,-1)
    for k in range(self.M_b):
      self.b_params[k].cores[self.root_node] = np.moveaxis(self.b_params[k].cores[self.root_node].squeeze(axis=-1), 0,-1)

    # init matrix and rhs variables
    A0_cores = [A_param.cores[A_param.tree.dim2id(0)] for A_param in self.A_params]
    b0_cores = [b_param.cores[b_param.tree.dim2id(0)] for b_param in self.b_params]
    self.A0 = assem_solve_fun.matrix(A0_cores)
    self.F0 = [np.hstack(F0k) for F0k in assem_solve_fun.rhs(b0_cores)]
    self.Nx = self.A0[0][0].shape[1]

    if rinit > 0:
      self.u = TreeBasedTensor.randn(tree=self.tree, shape=(self.Nx,) + self.shape, rank=rinit,
                                     dtype=self.dtype, seed=int(self.rng.integers(2**63)))
      self.u, indexset_list, indexset_dims_list, maxvol_ind_list = self._orth_towards_0(self.u)
      self.u.cores[self.root_node] = np.moveaxis(np.squeeze(self.u.cores[self.root_node], axis=-1), 0,-1)
    else:
      cores = NodeIndexedList([self.rng.standard_normal(c.shape) for c in self.A_params[0].cores])
      cores[self.tree.dim2leaf(0)] = self.rng.standard_normal((self.Nx, cores[self.tree.dim2leaf(0)].shape[1]))
      self.u = TreeBasedTensor(cores, self.tree, dtype=self.dtype)

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

    for iteration in range(self.iter_total, self.iter_total + n_iter):
      if compute_change and self._prev is None:
        self._prev = self._with_norm(self.get_tensor())
      if self.verbose > 0:
        print(f'= swp={iteration}')
      #### special core
      special = self.tree.root.children[0]
      U_prev = self.u.cores[special]

      # construct coeff
      arg_A = [None] * self.M_A
      arg_b = [None] * self.M_b
      for k in range(self.M_A):
        arg_A[k] = np.tensordot(self.A_params[k].cores[special], self.UA[k][self.tree.root], axes=(-1,-1))
      for k in range(self.M_b):
        arg_b[k] = np.tensordot(self.b_params[k].cores[special], self.Ub[k][self.tree.root], axes=(-1,-1))

      U0 = self.assem_solve_fun.solve(arg_A, arg_b)
      U0 = np.hstack(U0)
      self.n_eval += U0.shape[1]

      dx = 1
      if U_prev is not None:
        dx = np.linalg.norm(U0 - U_prev) / np.linalg.norm(U0)

      self.max_dx = max(self.max_dx, dx)
      if self.verbose > 1:
        print(f'    node {special.id} (s)'.ljust(20), f'dx={dx:.2e}')

      # truncate U0
      # TODO maybe use cheaper option
      U0, s,v = svd_cut(U0, tol/np.sqrt(self.tree.order))
      v = s[:, None] * v
      self.u.cores[special] = U0
      # cast non-orth factor to next core, which is root as child
      self.u.cores[self.root_node] = np.tensordot(self.u.cores[self.tree.root], v, axes=(-1,-1))

      ## rank adaption
      if self.kickrank > 0:
        # compute residual at indices
        rz = self.ZA[0][self.root_node].shape[0]
        Z0 = np.zeros((self.Nx, rz), dtype=self.dtype)
        cru = np.tensordot(U0 @ v, self.ZU[self.root_node], axes=(-1,-1))
        for k in range(self.M_A):
          for l in range(rz):
            if issparse(self.A0[k][0]):
              crA = csr_matrix((self.Nx, self.Nx), dtype=self.dtype)
            else:
              crA = np.zeros((self.Nx, self.Nx), dtype=self.dtype)
            for j, A0_j in enumerate(self.A0[k]):
              crA += A0_j * self.ZA[k][self.root_node][l,j]

            Z0[:,l] += np.ravel(crA @ cru[:,l])

        for k in range(self.M_b):
          Z0 -= np.tensordot(self.F0[k], self.Zb[k][self.root_node], axes=(-1,-1))

        # QR residual
        Z0 = np.linalg.qr(Z0)[0]
        # append residual to U core
        cru = np.hstack((U0, Z0))

        # QR enriched core
        U0, v = np.linalg.qr(cru)
        self.u.cores[special] = U0
        # cast non-orth factor to next core, which is root as child
        ru = self.u.cores[self.root_node].shape[-1]
        self.u.cores[self.root_node] = np.tensordot(self.u.cores[self.root_node], v[:, :ru], axes=(-1,-1))

      ## projection onto solution basis U0
      for k in range(self.M_A):
        proj = []
        for j, A0_j in enumerate(self.A0[k]):
          proj += [np.conjugate(U0.T) @ A0_j @ U0]

        self.UAU[k][special] = np.stack(proj, axis=-1)

      for k in range(self.M_b):
        self.UF[k][special] = (np.conjugate(U0.T) @ self.F0[k])

      ## Project onto residual
      if self.kickrank > 0:
        for k in range(self.M_A):
          proj = []
          for j, A0_j in enumerate(self.A0[k]):
            proj += [np.conjugate(Z0.T) @ self.A0[k][j] @ U0]

          self.ZUA[k][special] = np.stack(proj, axis=-1)

        for k in range(self.M_b):
          self.ZUb[k][special] = np.conjugate(Z0.T) @ self.F0[k]

      ## traverse tree
      self._als_interior_worker(self.root_node)

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

  @staticmethod
  def _inner(x: TreeBasedTensor, y: TreeBasedTensor) -> float:
    """
    Real part of sum(conj(x) * y) over all entries. Every subtree is contracted to an (r_x, r_y) matrix,
    which is applied to the child axis of the core of y first, so a node costs about r^(k+2) for k children
    (TreeBasedTensor.dot forms the Kronecker product of the cores, r^(2k+2) entries).
    """
    def worker(node):
      cx, cy = x.cores[node].conj(), y.cores[node]
      if node.isleaf:
        return cx.T @ cy
      k = node.n_children
      for i, child in enumerate(node.children):
        M = worker(child)
        cy = np.moveaxis(np.tensordot(M, cy, axes=([1], [i])), 0, i)
      return np.tensordot(cx, cy, axes=(list(range(k)), list(range(k))))
    return float(np.real(worker(x.tree.root)).sum())

  def _with_norm(self, u):
    return u, self._inner(u, u)

  def _change(self):
    """ Relative change ||u - u_prev|| / ||u|| of the solution since the last tracked sweep, which becomes u_prev. """
    (u, uu), (v, vv) = self._with_norm(self.get_tensor()), self._prev
    self._prev = (u, uu)
    if uu == 0: return float('inf')
    return float(np.sqrt(max(uu + vv - 2 * self._inner(u, v), 0.) / uu))

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
    root_core = np.expand_dims(np.moveaxis(root_core, -1, 0), -1)
    tensor.cores[tensor.tree.root] = root_core
    return tensor

  def _als_interior_worker(self, node: TreeNode):
    #### solve reduced system
    """
    Recursive sweep through the subtree below an interior node.

    Solves for the core of node, then for each child truncates (and enriches) the
    core towards the child, updates the projections from above and recurses. Finally
    solves again, truncates towards the parent, selects maxvol indices and updates
    the interpolation (UA, Ub) and projections (UAU, UF) of the subtree.

    Parameters
    ----------
    node: TreeNode
      Interior node of the modified tree (root_node or below).
    """
    dx = self._solve_reduced(node)
    self.max_dx = max(self.max_dx, dx)
    if self.verbose > 1:
      print(f'    node {node.id} down'.ljust(20), f'dx={dx:.2e}')

    #### Iterate over children
    for child_ind, child in enumerate(node.children):

      # orth and truncate solution core towards child
      core = self.u.cores[node]
      core = np.moveaxis(core, child_ind, -1)
      old_shape = core.shape
      core = core.reshape(-1, core.shape[-1])
      cru,s,v = svd_cut(core, tol=self.tol/np.sqrt(self.tree.order))
      # cast non orth factor to child
      v = s[:, None] * v
      self.u.cores[child] = np.tensordot(self.u.cores[child], v, axes=(-1,-1))

      ## AMEn rank adaption
      if self.kickrank > 0:
        U = (cru @ v).reshape(old_shape[:-1] + (-1,))
        U = np.moveaxis(U, -1, child_ind)
        crz = np.zeros(1, dtype=self.dtype)
        crz_new = np.zeros(1, dtype=self.dtype)
        # Au at res indices
        for k in range(self.M_A):
          crC = self.A_params[k].cores[node]

          # current
          offset = (node.n_children+1)
          einsum_args = [
            U, np.arange(1, 3*offset, 3),
            crC, np.arange(2, 3*offset, 3),
            self.ZU[child], [3*child_ind, 1+3*child_ind],
            self.ZA[k][child], [3*child_ind, 2+3*child_ind]
            ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.UAU[k][sibling], np.arange(3*si, 3*(si+1))]

          einsum_args += [self.UAU[k][node.parent], np.arange(3*(offset-1), 3*offset)]
          einsum_args += [np.arange(0,3*offset,3)]
          crz = crz + np.einsum(*einsum_args, optimize=True)

          # update
          einsum_args = [
            U, np.arange(1, 3*offset, 3),
            crC, np.arange(2, 3*offset, 3),
            self.ZU[child], [3*child_ind, 1+3*child_ind],
            self.ZA[k][child], [3*child_ind, 2+3*child_ind]
            ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.ZUA[k][sibling], np.arange(3*si, 3*(si+1))]

          einsum_args += [self.ZUA[k][node.parent], np.arange(3*(offset-1), 3*offset)]
          einsum_args += [np.arange(0,3*offset,3)]
          crz_new = crz_new + np.einsum(*einsum_args, optimize=True)

        # and corresponding RHS
        for k in range(self.M_b):
          crC = self.b_params[k].cores[node]

          # current residual
          offset = node.n_children+1
          einsum_args = [
            crC, np.arange(1, 2*offset, 2),
            self.Zb[k][child], [2*child_ind, 1+2*child_ind]
            ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.UF[k][sibling], [2*si, 2*si+1]]

          einsum_args += [self.UF[k][node.parent], [2*offset-2, 2*offset-1]]
          einsum_args += [np.arange(0, 2*offset, 2)]
          crz -= np.einsum(*einsum_args, optimize=True)

          # update residual
          einsum_args = [
            crC, np.arange(1, 2*offset, 2),
            self.Zb[k][child], [2*child_ind, 1+2*child_ind]
            ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.ZUb[k][sibling], [2*si, 2*si+1]]

          einsum_args += [self.ZUb[k][node.parent], [2*offset-2, 2*offset-1]]
          einsum_args += [np.arange(0, 2*offset, 2)]
          crz_new -= np.einsum(*einsum_args, optimize=True)

        # enrich by combining solution and residual
        crz = np.moveaxis(crz, child_ind, -1)
        crz = crz.reshape(-1, crz.shape[-1])
        cru = np.concatenate((cru, crz), axis=-1)
        cru, v = np.linalg.qr(cru)
        # cast non orth factor to child
        ru = self.u.cores[child].shape[-1]
        self.u.cores[child] = np.tensordot(self.u.cores[child], v[:, :ru], axes=(-1,-1))

      ## update core
      core = cru.reshape(old_shape[:-1] + (-1,))
      self.u.cores[node] = np.moveaxis(core, -1, child_ind)

      ## update interface projections
      cru = self.u.cores[node]
      cru_conj = np.conj(cru)

      # matrix projections
      for k in range(self.M_A):
        crC = self.A_params[k].cores[node]

        # build args list for einsum
        ind_end = 3*(node.n_children+1)
        einsum_args = [
          cru_conj, np.arange(0, ind_end, 3),
          cru, np.arange(1, ind_end, 3),
          crC, np.arange(2, ind_end, 3)
          ]
        for sibling in child.siblings:
          si = sibling.child_ind
          einsum_args += [self.UAU[k][sibling], np.arange(3*si, 3*(si+1))]

        einsum_args += [self.UAU[k][node.parent], np.arange(ind_end-3, ind_end)]
        einsum_args += [np.arange(3*child_ind, 3*(child_ind+1))]
        self.UAU[k][node] = np.einsum(*einsum_args, optimize=True)

      # RHS projections
      for k in range(self.M_b):
        crC = self.b_params[k].cores[node]

        ind_end = 2*(node.n_children+1)
        einsum_args = [
          cru_conj, np.arange(0, ind_end, 2),
          crC, np.arange(1, ind_end, 2)
          ]
        for sibling in child.siblings:
          si = sibling.child_ind
          einsum_args += [self.UF[k][sibling], np.arange(2*si, 2*(si+1))]

        einsum_args += [self.UF[k][node.parent], np.arange(ind_end-2, ind_end)]
        einsum_args += [np.arange(2*child_ind, 2*(child_ind+1))]

        self.UF[k][node] = np.einsum(*einsum_args, optimize=True)

      ## projections with the residual
      if self.kickrank > 0:
        crz_new = np.moveaxis(crz_new, child_ind, -1)
        old_shape = crz_new.shape[:-1]
        crz_new = crz_new.reshape(-1, crz_new.shape[-1])
        crz_new = np.linalg.qr(crz_new)[0]
        crz_new = crz_new.reshape(old_shape + (-1,))
        crz_new = np.moveaxis(crz_new, -1, child_ind)

        crz_new_conj = np.conjugate(crz_new)

        for k in range(self.M_A):
          ind_end = 3*(node.n_children+1)
          einsum_args = [
            crz_new_conj, np.arange(0, ind_end, 3),
            self.u.cores[node], np.arange(1, ind_end, 3),
            self.A_params[k].cores[node], np.arange(2, ind_end, 3),
          ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.ZUA[k][sibling], np.arange(3*si, 3*(si+1))]

          einsum_args += [self.ZUA[k][node.parent], np.arange(ind_end-3, ind_end)]
          einsum_args += [np.arange(3*child_ind, 3*(child_ind+1))]
          self.ZUA[k][node] = np.einsum(*einsum_args, optimize=True)

        for k in range(self.M_b):
          crC = self.b_params[k].cores[node]

          ind_end = 2*(node.n_children+1)
          einsum_args = [
            crz_new_conj, np.arange(0, ind_end, 2),
            crC, np.arange(1, ind_end, 2)
            ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.ZUb[k][sibling], np.arange(2*si, 2*(si+1))]

          einsum_args += [self.ZUb[k][node.parent], np.arange(ind_end-2, ind_end)]
          einsum_args += [np.arange(2*child_ind, 2*(child_ind+1))]

          self.ZUb[k][node] = np.einsum(*einsum_args, optimize=True)

        # TODO update Zb, ZA ?

      ## go to child
      if child.isleaf:
        self._als_leaf_worker(child)
      else:
        self._als_interior_worker(child)

    #### upwards move
    # solve
    dx = self._solve_reduced(node)
    self.max_dx = max(self.max_dx, dx)
    if self.verbose > 1:
      print(f'    node {node.id} up'.ljust(20), f'dx={dx:.2e}')

    # orth and truncate towards parent
    core = self.u.cores[node]
    old_shape = core.shape
    core = core.reshape(-1, core.shape[-1])
    cru, s, v = svd_cut(core, tol=self.tol/np.sqrt(self.tree.order))
    v = s[:, None] * v

    ## rank adaption
    if self.kickrank > 0:
      U = (cru @ v).reshape(old_shape[:-1] + (-1,))
      crz = np.zeros(1, dtype=self.dtype)
      crz_new = np.zeros(1, dtype=self.dtype)

      for k in range(self.M_A):
        crC = self.A_params[k].cores[node]

        # current
        offset = (node.n_children+1)
        einsum_args = [
          U, np.arange(offset),
          crC, np.arange(offset, 2*offset)
          ]
        for ci, child in enumerate(node.children):
          einsum_args += [self.UA[k][child], [ci, offset+ci]]

        einsum_args += [self.ZUA[k][node.parent], [2*offset, offset-1, 2*offset-1]]
        einsum_args += [np.concatenate([np.arange(offset-1), [2*offset]])]
        crz = crz + np.einsum(*einsum_args, optimize=True)

        # update
        einsum_args = [
          U, np.arange(1, 3*offset, 3),
          crC, np.arange(2, 3*offset, 3),
          ]
        for ci, child in enumerate(node.children):
          einsum_args += [self.ZU[child], [3*ci, 1+3*ci]]
          einsum_args += [self.ZA[k][child], [3*ci, 2+3*ci]]

        einsum_args += [self.ZUA[k][node.parent], np.arange(3*(offset-1), 3*offset)]
        einsum_args += [np.arange(0,3*offset,3)]
        crz_new = crz_new + np.einsum(*einsum_args, optimize=True)

      for k in range(self.M_b):
        crC = self.b_params[k].cores[node]

        # current residual
        offset = node.n_children+1
        einsum_args = [crC, np.arange(1, 2*offset, 2)]
        for ci, child in enumerate(node.children):
          einsum_args += [self.Ub[k][child], [2*ci, 2*ci+1]]

        einsum_args += [self.ZUb[k][node.parent], [2*offset-2, 2*offset-1]]
        einsum_args += [np.arange(0, 2*offset, 2)]
        crz -= np.einsum(*einsum_args, optimize=True)

        # update residual
        einsum_args = [crC, np.arange(1, 2*offset, 2)]
        for ci, child in enumerate(node.children):
          einsum_args += [self.Zb[k][child], [2*ci, 1+2*ci]]

        einsum_args += [self.ZUb[k][node.parent], [2*offset-2, 2*offset-1]]
        einsum_args += [np.arange(0, 2*offset, 2)]
        crz_new -= np.einsum(*einsum_args, optimize=True)

      # enrich core
      ru = cru.shape[-1]
      crz = crz.reshape(-1, crz.shape[-1])
      cru = np.concatenate((cru, crz), axis=-1)
      # orth
      cru, rv = np.linalg.qr(cru)
      v = rv[:,:ru] @ v


    # maxvol
    ind, C = rect_maxvol(cru, maxK=cru.shape[1], warn_budget=False)
    C = C.astype(self.dtype, copy=False)
    qmax = cru[ind]

    # update core
    self.u.cores[node] = C.reshape(old_shape[:-1] + (-1,))

    # cast non-orth factor to parent
    # TODO get rid of this hack, this stems from the tree modification
    if node == self.root_node:
      ci = 1
    else:
      ci = node.child_ind

    core = np.tensordot(self.u.cores[node.parent], qmax @ v, axes=(ci, -1))
    self.u.cores[node.parent] = np.moveaxis(core, -1, ci)

    ## update right interface projection (sample param on U indices)
    for k in range(self.M_A):
      tmp = self.A_params[k].cores[node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.UA[k][child], axes=(0, -1))

      tmp = np.moveaxis(tmp, 0, -1)
      tmp = tmp.reshape(-1, tmp.shape[-1])
      self.UA[k][node] = tmp[ind]

    for k in range(self.M_b):
      tmp = self.b_params[k].cores[node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.Ub[k][child], axes=(0, -1))

      tmp = np.moveaxis(tmp, 0, -1)
      tmp = tmp.reshape(-1, tmp.shape[-1])
      self.Ub[k][node] = tmp[ind]

    ## update left interface projection
    cru = self.u.cores[node]
    cru_conj = np.conj(cru)

    # Matrix projections
    for k in range(self.M_A):
      crC = self.A_params[k].cores[node]

      # build args list for einsum
      ind_end = 3*(node.n_children+1)
      einsum_args = [
        cru_conj, np.arange(0, ind_end, 3),
        cru, np.arange(1, ind_end, 3),
        crC, np.arange(2, ind_end, 3)
        ]
      for i, child in enumerate(node.children):
        einsum_args += [self.UAU[k][child], np.arange(3*i, 3*(i+1))]

      einsum_args += [np.arange(ind_end - 3, ind_end)]
      self.UAU[k][node] = np.einsum(*einsum_args, optimize=True)

    # RHS projections
    for k in range(self.M_b):
      crC = self.b_params[k].cores[node]

      ind_end = 2*(node.n_children+1)
      einsum_args = [
        cru_conj, np.arange(0, ind_end, 2),
        crC, np.arange(1, ind_end, 2)
        ]
      for i, child in enumerate(node.children):
        einsum_args += [self.UF[k][child], np.arange(2*i, 2*(i+1))]

      einsum_args += [np.arange(ind_end - 2, ind_end)]
      self.UF[k][node] = np.einsum(*einsum_args, optimize=True)

    # TODO update index set (for index based assem_solve_fun)

    ## sample at res indices
    if self.kickrank > 0:
      old_shape = crz_new.shape
      crz_new = crz_new.reshape(-1, crz_new.shape[-1])
      crz_new = np.linalg.qr(crz_new)[0]

      crz_new_conj = np.conjugate(crz_new).reshape(old_shape)

      #sample at res indices
      ind, C = rect_maxvol(crz_new, maxK=crz_new.shape[1], warn_budget=False)

      offset = node.n_children+1
      einsum_args = [self.u.cores[node], np.arange(offset, 2*offset)]
      for ci, child in enumerate(node.children):
        einsum_args += [self.ZU[child], [ci, offset+ci]]

      einsum_args += [np.concatenate((np.arange(offset-1), [2*offset-1]))]
      ZU = np.einsum(*einsum_args, optimize=True)
      ZU = ZU.reshape(-1, ZU.shape[-1])
      self.ZU[node] = ZU[ind]

      for k in range(self.M_A):
        crC = self.A_params[k].cores[node]

        einsum_args = [
          crz_new_conj, np.arange(0, 3*offset, 3),
          self.u.cores[node], np.arange(1, 3*offset, 3),
          crC, np.arange(2, 3*offset, 3),
        ]
        for ci, child in enumerate(node.children):
          einsum_args += [self.ZUA[k][child], np.arange(3*ci, 3*(ci+1))]

        einsum_args += [np.arange(3*offset-3, 3*offset)]
        self.ZUA[k][node] = np.einsum(*einsum_args, optimize=True)

        einsum_args = [crC, np.arange(offset, 2*offset)]
        for ci, child in enumerate(node.children):
          einsum_args += [self.ZA[k][child], [ci, offset+ci]]

        einsum_args += [np.concatenate((np.arange(offset-1), [2*offset-1]))]
        ZA = np.einsum(*einsum_args, optimize=True)
        ZA = ZA.reshape(-1, ZA.shape[-1])
        self.ZA[k][node] = ZA[ind]

      for k in range(self.M_b):
        crC = self.b_params[k].cores[node]

        einsum_args = [
          crz_new_conj, np.arange(0, 2*offset, 2),
          crC, np.arange(1, 2*offset, 2)
          ]
        for ci, child in enumerate(node.children):
          einsum_args += [self.ZUb[k][child], [2*ci, 1+2*ci]]

        einsum_args += [np.arange(2*offset-2, 2*offset)]
        self.ZUb[k][node] = np.einsum(*einsum_args, optimize=True)

        einsum_args = [crC, np.arange(offset, 2*offset)]
        for ci, child in enumerate(node.children):
          einsum_args += [self.Zb[k][child], [ci, offset+ci]]

        einsum_args += [np.concatenate((np.arange(offset-1), [2*offset-1]))]
        Zb = np.einsum(*einsum_args, optimize=True)
        Zb = Zb.reshape(-1, Zb.shape[-1])
        self.Zb[k][node] = Zb[ind]


  def _als_leaf_worker(self, node: TreeNode):
    # compute RHS projection
    """
    Solve for the core of a parameter leaf, truncate (and enrich) it, select maxvol
    indices and update the interpolation (UA, Ub) and projections (UAU, UF) of the leaf.

    Parameters
    ----------
    node: TreeNode
      Leaf of the modified tree, not the spatial leaf.
    """
    crF = np.zeros(1, dtype=self.dtype)
    for k in range(self.M_b):
      crC = self.b_params[k].cores[node]
      crF = crF + np.tensordot(crC, self.UF[k][node.parent], axes=(1,-1))

    # assemble and solve blocks
    cru = []
    for j in range(self.u.shape[node.dim]):
      Ai = np.zeros(1, dtype=self.dtype)
      for k in range(self.M_A):
        crC = self.A_params[k].cores[node]
        Ai = Ai + np.tensordot(self.UAU[k][node.parent], crC[j], axes=(-1, 0))

      cru += [np.linalg.solve(Ai, crF[j])]

    cru = np.hstack(cru)

    # check error
    dx = np.linalg.norm(cru.flatten() - self.u.cores[node].flatten()) / np.linalg.norm(cru)
    self.max_dx = max(self.max_dx, dx)
    if self.verbose > 1:
      print(f'    node {node.id} leaf'.ljust(20), f'dx={dx:.2e}')

    # update solution
    self.u.cores[node] = cru.reshape(self.u.cores[node].shape)

    # orth and truncate
    core = self.u.cores[node]
    old_shape = core.shape[:-1]
    cru, s, v = svd_cut(core, tol=self.tol/np.sqrt(self.tree.order))
    v = s[:, None] * v

    ## rank adaption
    if self.kickrank > 0:
      U = (cru @ v).reshape(old_shape + (-1,))
      crz = np.zeros(1, dtype=self.dtype)

      for k in range(self.M_A):
        einsum_args = [
          U, [0,1],
          self.A_params[k].cores[node], [0,2],
          self.ZUA[k][node.parent], [3,1,2],
          [0,3]
          ]
        crz = crz + np.einsum(*einsum_args, optimize=True)

      for k in range(self.M_b):
        einsum_args = [
          self.b_params[k].cores[node], [0,1],
          self.ZUb[k][node.parent], [2,1],
          [0,2]
          ]
        crz -= np.einsum(*einsum_args, optimize=True)

      # enrich core
      ru = cru.shape[-1]
      crz = crz.reshape(-1, crz.shape[-1])
      cru = np.concatenate((cru, crz), axis=-1)
      # orth
      cru, rv = np.linalg.qr(cru)
      v = rv[:,:ru] @ v

    # maxvol
    ind, C = rect_maxvol(cru, maxK=cru.shape[1], warn_budget=False)
    C = C.astype(self.dtype, copy=False)
    qmax = cru[ind]

    # update core
    self.u.cores[node] = C.reshape(old_shape + (-1,))

    # cast non-orth factor to parent
    # (child_ind refers to the modified tree, i.e. is already shifted for children of root_node)
    ci = node.child_ind
    core = self.u.cores[node.parent]
    core = np.tensordot(core, qmax @ v, axes=(ci, -1))
    self.u.cores[node.parent] = np.moveaxis(core, -1, ci)

    # update right interface projection (sample param on U indices)
    for k in range(self.M_A):
      self.UA[k][node] = self.A_params[k].cores[node][ind]

    for k in range(self.M_b):
      self.Ub[k][node] = self.b_params[k].cores[node][ind]

    # update left interface projection
    for k in range(self.M_A):
      cru = self.u.cores[node]
      crC = self.A_params[k].cores[node]
      self.UAU[k][node] = np.einsum('ab,ac,ad->bcd', np.conjugate(cru), cru, crC)

    for k in range(self.M_b):
      cru = self.u.cores[node]
      crC = self.b_params[k].cores[node]

      self.UF[k][node] = np.tensordot(np.conj(cru), crC, axes=(0,0))

    ## rank adaption
    if self.kickrank > 0:
      crz = np.linalg.qr(crz)[0]

      crz_new_conj = np.conjugate(crz)

      for k in range(self.M_A):
        einsum_args = [
          crz_new_conj, [0,1],
          self.u.cores[node], [0,2],
          self.A_params[k].cores[node], [0,3],
          [1,2,3]
        ]
        self.ZUA[k][node] = np.einsum(*einsum_args, optimize=True)

      for k in range(self.M_b):
        einsum_args = [
          crz_new_conj, [0,1],
          self.b_params[k].cores[node], [0,2],
          [1,2]
          ]
        self.ZUb[k][node] = np.einsum(*einsum_args, optimize=True)

      ind, C = rect_maxvol(crz, maxK=crz.shape[1], warn_budget=False)

      self.ZU[node] = self.u.cores[node][ind]

      for k in range(self.M_A):
        self.ZA[k][node] = self.A_params[k].cores[node][ind]

      for k in range(self.M_b):
        self.Zb[k][node] = self.b_params[k].cores[node][ind]


  def _solve_reduced(self, node:TreeNode):
    """
    Solve the local system for the core of an interior node and store it in u.

    The children are interpolated at their index sets (UA, Ub), the parent side is
    projected onto the solution basis (UAU, UF). This gives one small system per
    combination of child indices.

    Parameters
    ----------
    node: TreeNode
      Interior node of the modified tree.

    Returns
    -------
    dx: float
      Relative change of the core.
    """
    crA = [None] * self.M_A
    for k in range(self.M_A):
      tmp = self.A_params[k].cores[node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.UA[k][child], (0,-1))

      tmp = np.moveaxis(tmp, 0, -1)
      crA[k] = tmp.reshape(-1, tmp.shape[-1])

    # compute RHS projection
    crF = np.zeros(1, dtype=self.dtype)
    for k in range(self.M_b):
      tmp = self.b_params[k].cores[node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.Ub[k][child], (0,-1))

      crF = crF + np.tensordot(tmp, self.UF[k][node.parent], axes=(0,-1))

    crF = crF.reshape(-1, crF.shape[-1])

    # assemble and solve blocks
    cru = []
    for j in range(crA[0].shape[0]):
      Ai = np.zeros(1, dtype=self.dtype)
      for k in range(self.M_A):
        Ai = Ai + np.tensordot(self.UAU[k][node.parent], crA[k][j], axes=(-1, 0))

      cru += [np.linalg.solve(Ai, crF[j])]

    cru = np.hstack(cru)

    # check error
    dx = np.linalg.norm(cru.flatten() - self.u.cores[node].flatten()) / np.linalg.norm(cru)

    # update solution
    self.u.cores[node] = cru.reshape(self.u.cores[node].shape)

    return dx


  def _init_right_projection(self):
    """
    Initialize the projections UAU[k][node] (conj(U), U, A_k) and UF[k][node]
    (conj(U), b_k) of all subtrees below root_node onto the initial solution basis.
    The conjugated basis is always in the first slot.
    """
    self.UAU = [NodeIndexedList(self.tree.n_nodes * [None]) for k in range(self.M_A)]
    self.UF = [NodeIndexedList(self.tree.n_nodes * [None]) for k in range(self.M_b)]

    def worker(node: TreeNode):
      if node.isleaf:
        for k in range(self.M_A):
          cru = self.u.cores[node]
          crC = self.A_params[k].cores[node]
          self.UAU[k][node] = np.einsum('ab,ac,ad->bcd', np.conjugate(cru), cru, crC)

        for k in range(self.M_b):
          cru = self.u.cores[node]
          crC = self.b_params[k].cores[node]

          self.UF[k][node] = np.tensordot(np.conj(cru), crC, axes=(0,0))

      else:
        cru = self.u.cores[node]
        cru_conj = np.conj(cru)

        # recurse to children
        for i, child in enumerate(node.children):
          worker(child)

        # Matrix projections
        for k in range(self.M_A):
          crC = self.A_params[k].cores[node]

          # build args list for einsum
          ind_end = 3*(node.n_children+1)
          einsum_args = [
            cru_conj, np.arange(0, ind_end, 3),
            cru, np.arange(1, ind_end, 3),
            crC, np.arange(2, ind_end, 3)
            ]
          for i, child in enumerate(node.children):
            einsum_args += [self.UAU[k][child], np.arange(3*i, 3*(i+1))]

          einsum_args += [np.arange(ind_end - 3, ind_end)]
          self.UAU[k][node] = np.einsum(*einsum_args, optimize=True)

        # RHS projections
        for k in range(self.M_b):
          crC = self.b_params[k].cores[node]

          ind_end = 2*(node.n_children+1)
          einsum_args = [
            cru_conj, np.arange(0, ind_end, 2),
            crC, np.arange(1, ind_end, 2)
            ]
          for i, child in enumerate(node.children):
            einsum_args += [self.UF[k][child], np.arange(2*i, 2*(i+1))]

          einsum_args += [np.arange(ind_end - 2, ind_end)]

          self.UF[k][node] = np.einsum(*einsum_args, optimize=True)

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
      if node.isleaf:
        partial_evals[node] = tensor.cores[node][maxvol_ind_list[node]]
        return
      else:
        # contract children in order, so that the first child is the slowest index
        # (as for the maxvol indices from _orth_subtree_maxvol)
        tmp = tensor.cores[node]
        for child in node.children:
          worker(child)
          tmp = np.tensordot(tmp, partial_evals[child], axes=(0, -1))

        tmp = np.moveaxis(tmp, 0, -1)
        tmp = tmp.reshape(-1, tmp.shape[-1])
        partial_evals[node] = tmp[maxvol_ind_list[node]]

    worker(self.root_node)
    return partial_evals

  def _init_AMEn_data(self):
    # no solution yet, fill random
    """
    Initialize the data for the residual enrichment: the solution and parameters at
    random residual indices (ZU, ZA, Zb) and random residual projections (ZUA, ZUb).
    """
    self.ZUA = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.A_params] # rz x ru x rA
    self.ZUb = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.b_params] # rz x rb
    # init at random indices
    self.ZU = NodeIndexedList(self.tree.n_nodes * [None]) # rz x ru
    self.ZA = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.A_params] # rz x rA
    self.Zb = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.b_params] # rz x rb

    def worker(node):
      if node.isleaf:
        n, ru = self.u.cores[node].shape
        ind = self.rng.choice(np.arange(n), self.kickrank, replace=True) #TODO replace=False?
        self.ZU[node] = self.u.cores[node][ind]

        for k in range(self.M_A):
          rA = self.A_params[k].cores[node].shape[-1]
          self.ZUA[k][node] = self.rng.standard_normal((self.kickrank, ru, rA)).astype(self.dtype)
          self.ZA[k][node] = self.A_params[k].cores[node][ind]

        for k in range(self.M_b):
          rb = self.b_params[k].cores[node].shape[-1]
          self.ZUb[k][node] = self.rng.standard_normal((self.kickrank, rb)).astype(self.dtype)
          self.Zb[k][node] = self.b_params[k].cores[node][ind]

      else:
        for child in node.children:
          worker(child)

        ru = self.u.cores[node].shape[-1]
        r_add = min(ru, self.kickrank)
        indices = np.arange(np.prod([self.ZU[child].shape[0] for child in node.children]))
        ind = self.rng.choice(indices, r_add, replace=True) #TODO replace=False?

        offset = node.n_children + 1
        einsum_args = [self.u.cores[node], np.arange(offset, 2*offset)]
        for ci, child in enumerate(node.children):
          einsum_args += [self.ZU[child], [ci, offset+ci]]

        einsum_args += [np.concatenate([np.arange(offset-1), [2*offset-1]])]
        ZU = np.einsum(*einsum_args, optimize=True)
        ZU = ZU.reshape(-1, ZU.shape[-1])
        self.ZU[node] = ZU[ind]

        for k in range(self.M_A):
          rA = self.A_params[k].cores[node].shape[-1]

          self.ZUA[k][node] = self.rng.standard_normal((r_add, ru, rA)).astype(self.dtype)

          crC = self.A_params[k].cores[node]
          offset = node.n_children + 1
          einsum_args = [crC, np.arange(offset, 2*offset)]
          for ci, child in enumerate(node.children):
            einsum_args += [self.ZA[k][child], [ci, offset+ci]]

          einsum_args += [np.concatenate([np.arange(offset-1), [2*offset-1]])]
          ZA = np.einsum(*einsum_args, optimize=True)
          ZA = ZA.reshape(-1, ZA.shape[-1])
          self.ZA[k][node] = ZA[ind]

        for k in range(self.M_b):
          rb = self.b_params[k].cores[node].shape[-1]
          self.ZUb[k][node] = self.rng.standard_normal((r_add, rb)).astype(self.dtype)

          crC = self.b_params[k].cores[node]
          offset = node.n_children + 1
          einsum_args = [crC, np.arange(offset, 2*offset)]
          for ci, child in enumerate(node.children):
            einsum_args += [self.Zb[k][child], [ci, offset+ci]]

          einsum_args += [np.concatenate([np.arange(offset-1), [2*offset-1]])]
          Zb = np.einsum(*einsum_args, optimize=True)
          Zb = Zb.reshape(-1, Zb.shape[-1])
          self.Zb[k][node] = Zb[ind]

    worker(self.root_node)


  @staticmethod
  def _orth_towards_0(tensor: TreeBasedTensor, return_indices = True):

    # orth towards root
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
    _, indexset_list, indexset_dims_list, maxvol_ind_list = tensor._orth_subtree_maxvol(tensor.tree.root)

    # assume dim 0 lives in first child of root
    # orth towards dim 0
    core = tensor.cores[tensor.tree.root]
    old_shape = core.shape
    core = core.reshape(core.shape[0], -1)
    q,r = np.linalg.qr(core.T)
    ind, C = rect_maxvol(q, maxK=q.shape[-1], warn_budget=False)
    C = C.astype(tensor.dtype, copy=False)
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






