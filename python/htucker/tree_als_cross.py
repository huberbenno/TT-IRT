import numpy as np

from tree_tensor import TreeBasedTensor
from tree import NodeIndexedList
from maxvolpy.maxvol import rect_maxvol, svd_cut

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

    self.rng = np.random.default_rng()

    # tree and shape must be the same for all parameters
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
        
    # init matrix and rhs variables
    A0_cores = [A_param.cores[A_param.tree.dim2id(0)] for A_param in self.A_params]
    b0_cores = [b_param.cores[b_param.tree.dim2id(0)] for b_param in self.b_params]
    self.A0 = assem_solve_fun.matrix(A0_cores)
    self.F0 = [np.hstack(F0k) for F0k in assem_solve_fun.rhs(b0_cores)]

    self.Nx = self.A0[0][0].shape[1]

    self.UA = [self._partial_evals(A_param, maxvol_ind_list) for A_param in self.A_params]
    self.Ub = [self._partial_evals(b_param, maxvol_ind_list) for b_param in self.b_params]

    cores = NodeIndexedList([np.zeros_like(c) for c in self.A_params[0].cores])
    cores[self.tree.dim2leaf(0)] = np.zeros((self.Nx, cores[self.tree.dim2leaf(0)].shape[1]))
    self.u = TreeBasedTensor(cores, self.tree)

    self.UAU = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.A_params]
    self.UF = [NodeIndexedList(self.tree.n_nodes * [None]) for _ in self.b_params]

  def run(self, n_iter=1, tol=1e-3, verbose=False):
    max_dx = 0

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

      max_dx = max(max_dx, self.dx)

      if verbose > 0:
        print(f'= swp={iteration} core 0, max_dx={max_dx:.2e}')

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
         proj += [np.conjugate(U0) @ A0_j @ U0]
      
        self.UAU[k][special] = np.concatenate(proj, axis=-1)
      
      for k in range(self.M_b):
        self.UF[k][special] = (np.conjugate(U0.T) @ self.F0[k])

      #### root
      # projection of interface towards special
      root = self.tree.root
      crA = [None] * self.M_A
      for k in range(self.M_A):
        tmp = self.A_params[k][root].squeeze(-1)

        for child in root.children[1:]:
          tmp = np.tensordot(tmp, self.UA[k][child], (1,-1))
        
        crA[k] = tmp.reshape(tmp.shape[0], -1)

      # compute RHS projection
      crF = np.zeros(1)
      for k in range(self.M_b):
        tmp = self.b_params[k][root].squeeze(-1)
        for child in root.children[1:]:
          tmp = np.tensordot(tmp, self.Ub[k][child], (1,-1))
        
        crF += np.tensordot(self.UF[k][special], tmp, axes=(-1,0))

      # assemble and solve blocks
      cru = []
      for j in range(crA[0].shape[1]):
        Ai = np.zeros(1)
        for k in range(self.M_A):
          Ai += np.tensordot(self.UAU[k][special], crA[k][:,j], axes=(-1, 0))

        cru[:,j] = np.linalg.solve(Ai, crF[:,j])

      # check error
      dx = np.linalg.norm(cru.flatten() - self.u.cores[root].flatten()) / np.linalg.norm(cru)

      max_dx = max(max_dx, dx)

      # update solution
      self.u.cores[root] = cru.reshape(self.u.cores[root].shape)


      ####
      for child in root.children[1:]:
        child_ind = child.child_ind

        # orth and truncate solution core
        core = self.u.cores[root]
        core = np.moveaxis(core, child_ind, -1)
        old_shape = core.shape
        core = core.reshape(-1, core.shape[-1])
        cru,s,v = svd_cut(core, tol=tol/np.sqrt(self.tree.order))
        
        core = cru.reshape(old_shape[:-1] + (-1,))
        self.u.cores[root] = np.moveaxis(core, -1, child_ind)
        
        # cast non orth factor to child
        self.u.cores[child] = np.tensordot(self.u.cores[child], (np.diag(s) @ v), axes=(-1,-1))

        # update left interface projections
        for k in range(self.M_A):
          UAU = self.UAU[k][special]
          cru = self.u.cores[root]
          cru = np.moveaxis(cru.squeeze(-1), 0, -1)
          crC = self.A_params[k].cores[root]
          crC = np.moveaxis(crC.squeeze(-1), 0, -1)

          ind_offset = root.n_children - 1
          tmp = np.tensordot(cru, UAU, axes=(-1,1))
          tmp = np.tensordot(np.conjugate(cru), tmp, axes=(-1,-2))
          tmp = np.tensordot(tmp, crC, axes=(-1,-1))
          for s in child.siblings:
            if s.child_ind == 0: continue
            tmp = np.tensordot(tmp, self.UAU[k][s], axes=((0, ind_offset, 2*ind_offset),(0,1,2)))
            ind_offset -= 1

          self.UAU[k][root] = tmp

        # update RHS projection interfaces
        for k in range(self.M_b):
          UF = self.UF[k][special]
          cru = self.u.cores[root]
          cru = np.moveaxis(cru, child_ind, -1)
          crC = self.b_params[k].cores[root]
          crC = np.moveaxis(cru, child_ind, -1)

          ind_offset = root.n_children - 1
          tmp = np.tensordot(np.concatenate(cru), UAU, axes=(-1,0))
          tmp = np.tensordot(tmp, crC, axes=(-1,-1))
          for s in child.siblings:
            if s.child_ind == 0: continue
            tmp = np.tensordot(tmp, self.UF[k][s], axes=((0, ind_offset),(0,1)))
            ind_offset -= 1

          self.UF[k][root] = tmp

      #### Rest of the tree
      


  @staticmethod
  def _partial_evals(tensor: TreeBasedTensor, maxvol_ind_list):
    partial_evals = NodeIndexedList(tensor.tree.n_nodes * [None])

    def worker(node):
      if node.isleaf:
        partial_evals[node] = tensor.cores[node][maxvol_ind_list[node]]
        return
      else:
        if node.isroot:
          # special case: towards 0 child
          tmp = tensor.cores[node].squeeze(-1)
          for child in node.children[1:]:
            worker(child)
            tmp = np.tensordot(partial_evals[child], tmp, (-1,1))
        else:
          tmp = tensor.cores[node]
          for child in node.children:
            worker(child)
            tmp = np.tensordot(partial_evals[child], tmp, (-1,0))
          
        tmp = tmp.reshape(-1, tmp.shape[-1])
        partial_evals[node] = tmp[maxvol_ind_list[node]]

    worker(tensor.tree.root)
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
        indexset_dims += indexset_dims_list[child]
        indexset = np.hstack(
          np.repeat(indexset, repeats=indexset_list[child].shape[0], axis=0),
          np.tile(indexset_list[child], reps=(indexset.shape[0], 1))
        )

      # put the new indexset into root spot
      # TODO reconsider 
      indexset_list[tensor.tree.root] = indexset[ind]
      indexset_dims_list[tensor.tree.root] = indexset_list
      maxvol_ind_list[tensor.tree.root] = ind

      return tensor, indexset_list, indexset_dims_list, maxvol_ind_list
    else:
      return tensor






