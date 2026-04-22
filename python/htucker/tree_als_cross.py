import numpy as np

from tree_tensor import TreeBasedTensor
from tree import NodeIndexedList, TreeNode
from maxvolpy.maxvol import rect_maxvol, svd_cut
import copy

class TreeALSCross:
  def __init__(
      self,
      A_params: list[TreeBasedTensor],
      b_params: list[TreeBasedTensor],
      assem_solve_fun,
      use_indices=False
    ):
    # store parameters
    self.A_params = A_params
    self.b_params = b_params
    self.assem_solve_fun = assem_solve_fun

    self.verbose = True

    self.rng = np.random.default_rng()

    # tree and shape must be the same for all parameters (except spatial dim)
    # currently not checked
    self.tree = A_params[0].tree
    self.shape = A_params[0].shape

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
    self.root_node = copy.copy(self.tree.root)
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

    # init right proj UA and Ub (eval at maxvol)
    self.UA = [self._partial_evals(A_param, maxvol_ind_list) for A_param in self.A_params]
    self.Ub = [self._partial_evals(b_param, maxvol_ind_list) for b_param in self.b_params]

    cores = NodeIndexedList([np.zeros_like(c) for c in self.A_params[0].cores])
    cores[self.tree.dim2leaf(0)] = np.zeros((self.Nx, cores[self.tree.dim2leaf(0)].shape[1]))
    self.u = TreeBasedTensor(cores, self.tree)
    # self.u.cores[self.root_node] = np.squeeze(self.u.cores[self.root_node], axis=-1)

    # init UAU and UF
    self._init_right_projection()
    if self.verbose:
      print('Init finished')

  def run(self, n_iter=1, tol=1e-3, verbose=False):
    self.tol = tol
    self.max_dx = 0

    for iteration in range(n_iter):
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
        self.dx = np.linalg.norm(U0 - U_prev) / np.linalg.norm(U0)

      self.max_dx = max(self.max_dx, self.dx)

      if verbose > 0:
        print(f'= swp={iteration} core 0, max_dx={self.max_dx:.2e}')

      # truncate U0
      # TODO maybe use cheaper option
      U0, s,v = svd_cut(U0, tol/np.sqrt(self.tree.order))
      v = np.diag(s) @ v
      self.u.cores[special] = U0
      # cast non-orth factor to next core
      self.u.cores[self.tree.root] = np.tensordot(v, self.u.cores[self.tree.root], axes=(-1,0))

      # projection onto solution basis U0
      for k in range(self.M_A):
        proj = []
        for j, A0_j in enumerate(self.A0[k]):
          proj += [np.conjugate(U0.T) @ A0_j @ U0]

        self.UAU[k][special] = np.stack(proj, axis=-1)

      for k in range(self.M_b):
        self.UF[k][special] = (np.conjugate(U0.T) @ self.F0[k])

      self._als_interior_worker(self.root_node)

  def _als_interior_worker(self, node: TreeNode):
    #### solve reduced system
    dx = self._solve_reduced(node)
    self.max_dx = max(self.max_dx, dx)

    #### Iterate over children
    for child_ind, child in enumerate(node.children):

      # orth and truncate solution core towards child
      core = self.u.cores[node]
      core = np.moveaxis(core, child_ind, -1)
      old_shape = core.shape
      core = core.reshape(-1, core.shape[-1])
      cru,s,v = svd_cut(core, tol=self.tol/np.sqrt(self.tree.order))

      core = cru.reshape(old_shape[:-1] + (-1,))
      self.u.cores[node] = np.moveaxis(core, -1, child_ind)

      # cast non orth factor to child
      self.u.cores[child] = np.tensordot(self.u.cores[child], (np.diag(s) @ v), axes=(-1,-1))

      # update interface projections
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

        einsum_args += [self.UAU[k][node.parent], [ind_end-3, ind_end]]
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
          einsum_args += [self.UF[k][sibling], np.arange(2*si, 2*(si+1))]
        
        einsum_args += [self.UAU[k][node.parent], [ind_end-3, ind_end]]
        einsum_args += [np.arange(2*child_ind, 2*(child_ind+1))]

        self.UF[k][node] = np.einsum(*einsum_args, optimize=True)

      # go to child
      if child.isleaf:
        self._als_leaf_worker(child)
      else:
        self._als_interior_worker(child)

    #### upwards move
    # solve
    dx = self._solve_reduced_system(node)

    # orth and truncate towards parent
    core = self.u.cores[node]
    old_shape = core.shape
    core = core.reshape(-1, core.shape[0])
    cru, s, v = svd_cut(core, tol=self.tol/np.sqrt(self.tree.order))

    # maxvol
    ind, C = rect_maxvol(cru, maxK=cru.shape[1])
    qmax = cru[ind]

    # update core
    self.u.cores[node] = C.reshape(old_shape + (-1,))

    # cast non-orth factor to parent
    core = self.u.cores[node.parent]
    core = np.tensordot(core, qmax @ np.diag(s) @ v, axes=(child_ind, -1))
    self.u.cores[node.parent] = np.moveaxis(core, -1, child_ind)

    # update right interface projection (sample param on U indices)
    for k in range(self.M_A):
      tmp = self.A_params[k][node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.UA[k][child], axes=(0, -1))

      tmp = np.moveaxis(tmp, 0, -1)
      tmp = tmp.reshape(-1, tmp.shape[-1])
      self.UA[k][node] = tmp[ind]

    for k in range(self.M_b):
      tmp = self.b_params[k][node]
      for child in node.children:
        tmp = np.tensordot(tmp, self.Ub[k][child], axes=(0, -1))

      tmp = np.moveaxis(tmp, 0, -1)
      tmp = tmp.reshape(-1, tmp.shape[-1])
      self.Ub[k][node] = tmp[ind]

    # update left interface projection
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


  def _als_leaf_worker(self, node: TreeNode):
    # compute RHS projection
    crF = np.zeros(1)
    for k in range(self.M_b):
      tmp = self.b_params[k].cores[node]
      crF += np.tensordot(tmp, self.UF[k][node.parent], axes=(0,-1))

    # assemble and solve blocks
    cru = []
    for j in range(self.u.shape[node.id]):
      Ai = np.zeros(1)
      for k in range(self.M_A):
        Ai += np.tensordot(self.UAU[k][node.parent], self.A_params[k][node][j], axes=(-1, 0))

      cru += [np.linalg.solve(Ai, crF[j])]

    cru = np.hstack(cru)

    # check error
    dx = np.linalg.norm(cru.flatten() - self.u.cores[node].flatten()) / np.linalg.norm(cru)

    # update solution
    self.u.cores[node] = cru.reshape(self.u.cores[node].shape)

    # orth and truncate
    core = self.u.cores[node]
    old_shape = core.shape[:-1]
    cru, s, v = svd_cut(core, tol=self.tol/np.sqrt(self.tree.order))

    ind, C = rect_maxvol(cru, maxK=cru.shape[1])
    qmax = cru[ind]

    # update core
    self.u.cores[node] = C.reshape(old_shape + (-1,))

    # cast non-orth factor to parent
    core = self.u.cores[node.parent]
    core = np.tensordot(core, qmax @ np.diag(s) @ v, axes=(node.child_ind, -1))
    self.u.cores[node.parent] = np.moveaxis(core, -1, node.child_ind)

    # update right interface projection (sample param on U indices)
    for k in range(self.M_A):
      self.UA[k][node] = self.A_params[k][node][ind]

    for k in range(self.M_b):
      self.Ub[k][node] = self.b_params[k][node][ind]

    # update left interface projection
    for k in range(self.M_A):
      cru = self.u.cores[node]
      crC = self.A_params[k].cores[node]
      self.UAU[k][node] = np.einsum('ab,ac,ad->bcd', np.conjugate(cru), cru, crC)

    for k in range(self.M_b):
      cru = self.u.cores[node]
      crC = self.b_params[k].cores[node]

      self.UF[k][node] = np.tensordot(cru, crC, axes=(0,0))

    return dx


  def _solve_reduced(self, node:TreeNode):
    crA = [None] * self.M_A
    for k in range(self.M_A):
      tmp = self.A_params[k].cores[node]
      print('start', tmp.shape)
      for child in node.children:
        tmp = np.tensordot(tmp, self.UA[k][child], (0,-1))
      
      tmp = np.moveaxis(tmp, 0, -1)
      crA[k] = tmp.reshape(-1, tmp.shape[-1])
      print(crA[k].shape)

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






