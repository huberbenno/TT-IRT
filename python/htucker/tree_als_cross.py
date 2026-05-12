import numpy as np

from tree_tensor import TreeBasedTensor
from tree import NodeIndexedList, TreeNode
from maxvolpy.maxvol import rect_maxvol, svd_cut
import copy
from scipy.sparse import csr_matrix, issparse

class TreeALSCross:
  def __init__(
      self,
      A_params: list[TreeBasedTensor],
      b_params: list[TreeBasedTensor],
      assem_solve_fun,
      use_indices=False,
      kickrank=0,
      rinit=0,
      verbose=0
    ):
    # store parameters
    self.A_params = [copy.deepcopy(A_param) for A_param in A_params]
    self.b_params = [copy.deepcopy(b_param) for b_param in b_params]
    self.assem_solve_fun = assem_solve_fun

    self.verbose = verbose
    self.kickrank = kickrank

    self.rng = np.random.default_rng()

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
      self.u = TreeBasedTensor.randn(tree=self.tree, shape=(self.Nx,) + self.shape, rank=rinit)
      self.u, indexset_list, indexset_dims_list, maxvol_ind_list = self._orth_towards_0(self.u)
      self.u.cores[self.root_node] = np.moveaxis(np.squeeze(self.u.cores[self.root_node], axis=-1), 0,-1)
    else:
      cores = NodeIndexedList([self.rng.standard_normal(c.shape) for c in self.A_params[0].cores])
      cores[self.tree.dim2leaf(0)] = self.rng.standard_normal((self.Nx, cores[self.tree.dim2leaf(0)].shape[1]))
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

  def run(self, n_iter=1, tol=1e-3):
    self.tol = tol
    self.max_dx = 0

    for iteration in range(n_iter):
      if self.verbose > 0:
        print(f'= swp={iteration}')
      #### special core
      special = self.tree.root.children[0]
      U_prev = self.u.cores[special]

      # construct coeff
      arg = [[None] * self.M_A, [None] * self.M_b]
      for k in range(self.M_A):
        arg[0][k] = np.tensordot(self.A_params[k].cores[special], self.UA[k][self.tree.root], axes=(-1,-1))
      for k in range(self.M_b):
        arg[1][k] = np.tensordot(self.b_params[k].cores[special], self.Ub[k][self.tree.root], axes=(-1,-1))

      U0 = self.assem_solve_fun.solve(arg)
      U0 = np.hstack(U0)

      dx = 1
      if U_prev is not None:
        dx = np.linalg.norm(U0 - U_prev) / np.linalg.norm(U0)

      self.max_dx = max(self.max_dx, dx)
      if self.verbose > 1:
        print(f'    node {special.id} (s)'.ljust(20), f'dx={dx:.2e}')

      # truncate U0
      # TODO maybe use cheaper option
      U0, s,v = svd_cut(U0, tol/np.sqrt(self.tree.order))
      v = np.diag(s) @ v
      self.u.cores[special] = U0
      # cast non-orth factor to next core, which is root as child
      self.u.cores[self.root_node] = np.tensordot(self.u.cores[self.tree.root], v, axes=(-1,-1))

      ## rank adaption
      if self.kickrank > 0:
        # compute residual at indices
        rz = self.ZA[0][self.root_node].shape[0]
        Z0 = np.zeros((self.Nx, rz))
        for k in range(self.M_A):
          cru = np.tensordot(U0 @ v, self.ZU[self.root_node], axes=(-1,-1))
          for l in range(rz):
            if issparse(self.A0[k][0]):
              crA = csr_matrix((self.Nx, self.Nx))
            else:
              crA = np.zeros((self.Nx, self.Nx))
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

      if self.verbose > 0:
        print(f'= swp={iteration} finished, max_dx={self.max_dx:.2e}')
      self.max_dx = 0

  def get_tensor(self):
    tensor = TreeBasedTensor(self.u)
    root_core = tensor.cores[tensor.tree.root]
    root_core = np.expand_dims(np.moveaxis(root_core, -1, 0), -1)
    tensor.cores[tensor.tree.root] = root_core
    return tensor

  def _als_interior_worker(self, node: TreeNode):
    #### solve reduced system
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
      v = np.diag(s) @ v
      self.u.cores[child] = np.tensordot(self.u.cores[child], v, axes=(-1,-1))

      ## AMEn rank adaption
      if self.kickrank > 0:
        U = (cru @ v).reshape(old_shape[:-1] + (-1,))
        U = np.moveaxis(U, -1, child_ind)
        crz = np.zeros(1)
        crz_new = np.zeros(1)
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
            self.Zb[k][child], [1+2*child_ind, 2*child_ind]
            ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.UF[k][sibling], [2*si, 2*(si+1)]]

          einsum_args += [self.UF[k][node.parent], [2*(offset-1), 2*offset]]
          einsum_args += [np.arange(0, 2*offset, 2)]
          crz -= np.einsum(*einsum_args, optimize=True)

          # update residual
          einsum_args = [
            crC, np.arange(1, 2*offset, 2),
            self.Zb[k][child], [2*child_ind, 1+2*child_ind]
            ]
          for sibling in child.siblings:
            si = sibling.child_ind
            einsum_args += [self.ZUb[k][sibling], [2*si, 2*(si+1)]]

          einsum_args += [self.ZUb[k][node.parent], [2*(offset-1), 2*offset]]
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
          cru, np.arange(0, ind_end, 3),
          cru_conj, np.arange(1, ind_end, 3),
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
    v = np.diag(s) @ v

    ## rank adaption
    if self.kickrank > 0:
      U = (cru @ v).reshape(old_shape[:-1] + (-1,))
      crz = np.zeros(1)
      crz_new = np.zeros(1)

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
          einsum_args += [self.Ub[k][child], [2*ci, 2*(ci+1)]]

        einsum_args += [self.ZUb[k][node.parent], [2*(offset-1), 2*offset]]
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
    ind, C = rect_maxvol(cru, maxK=cru.shape[1])
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
        cru, np.arange(0, ind_end, 3),
        cru_conj, np.arange(1, ind_end, 3),
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
      ind, C = rect_maxvol(crz_new, maxK=crz_new.shape[1])

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
    crF = np.zeros(1)
    for k in range(self.M_b):
      crC = self.b_params[k].cores[node]
      crF = crF + np.tensordot(crC, self.UF[k][node.parent], axes=(1,-1))

    # assemble and solve blocks
    cru = []
    for j in range(self.u.shape[node.dim]):
      Ai = np.zeros(1)
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
    v = np.diag(s) @ v

    ## rank adaption
    if self.kickrank > 0:
      U = (cru @ v).reshape(old_shape + (-1,))
      crz = np.zeros(1)

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
    ind, C = rect_maxvol(cru, maxK=cru.shape[1])
    qmax = cru[ind]

    # update core
    self.u.cores[node] = C.reshape(old_shape + (-1,))

    # cast non-orth factor to parent
    if node.parent.id == self.root_node.id:
      ci = node.child_ind - 1
    else:
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

      self.UF[k][node] = np.tensordot(cru, crC, axes=(0,0))

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

      ind, C = rect_maxvol(crz, maxK=crz.shape[1])

      self.ZU[node] = self.u.cores[node][ind]

      for k in range(self.M_A):
        self.ZA[k][node] = self.A_params[k].cores[node][ind]

      for k in range(self.M_b):
        self.Zb[k][node] = self.b_params[k].cores[node][ind]


  def _solve_reduced(self, node:TreeNode):
    crA = [None] * self.M_A
    for k in range(self.M_A):
      tmp = self.A_params[k].cores[node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.UA[k][child], (0,-1))

      tmp = np.moveaxis(tmp, 0, -1)
      crA[k] = tmp.reshape(-1, tmp.shape[-1])

    # compute RHS projection
    crF = np.zeros(1)
    for k in range(self.M_b):
      tmp = self.b_params[k].cores[node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.Ub[k][child], (0,-1))

      crF = crF + np.tensordot(tmp, self.UF[k][node.parent], axes=(0,-1))

    crF = crF.reshape(-1, crF.shape[-1])

    # assemble and solve blocks
    cru = []
    for j in range(crA[0].shape[0]):
      Ai = np.zeros(1)
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

          self.UF[k][node] = np.tensordot(cru, crC, axes=(0,0))

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
            cru, np.arange(0, ind_end, 3),
            cru_conj, np.arange(1, ind_end, 3),
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
    partial_evals = NodeIndexedList(tensor.tree.n_nodes * [None])

    def worker(node):
      if node.isleaf:
        partial_evals[node] = tensor.cores[node][maxvol_ind_list[node]]
        return
      else:
        tmp = tensor.cores[node]
        for i, child in enumerate(node.children):
          worker(child)
          tmp = np.tensordot(partial_evals[child], tmp, (-1,i))

        tmp = tmp.reshape(-1, tmp.shape[-1])
        partial_evals[node] = tmp[maxvol_ind_list[node]]

    worker(self.root_node)
    return partial_evals

  def _init_AMEn_data(self):
    # no solution yet, fill random
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
          self.ZUA[k][node] = self.rng.standard_normal((self.kickrank, ru, rA))
          self.ZA[k][node] = self.A_params[k].cores[node][ind]

        for k in range(self.M_b):
          rb = self.b_params[k].cores[node].shape[-1]
          self.ZUb[k][node] = self.rng.standard_normal((self.kickrank, rb))
          self.Zb[k][node] = self.b_params[k].cores[node][ind]

      else:
        for child in node.children:
          worker(child)

        ru = self.u.cores[node].shape[-1]
        ind = self.rng.choice(np.arange(self.kickrank**node.n_children), self.kickrank, replace=True) #TODO replace=False?

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

          self.ZUA[k][node] = self.rng.standard_normal((self.kickrank, ru, rA))

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
          self.ZUb[k][node] = self.rng.standard_normal((self.kickrank, rb))

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
    _, indexset_list, indexset_dims_list, maxvol_ind_list = tensor._orth_subtree_maxvol(tensor.tree.root)

    # assume dim 0 lives in first child of root
    # orth towards dim 0
    core = tensor.cores[tensor.tree.root]
    old_shape = core.shape
    core = core.reshape(core.shape[0], -1)
    q,r = np.linalg.qr(core.T)
    ind, C = rect_maxvol(q, maxK=q.shape[-1])
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






