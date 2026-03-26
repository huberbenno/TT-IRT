import numpy as np
from maxvolpy.maxvol import maxvol, rect_maxvol, svd_cut
from localcross import localcross

class HTTree:
  '''
  Hierarchical Tucker tensor format object.
  '''
  def __init__(self):
    self.root = None
    self.d = None

  def setup(self):
    self.d = self.root._count_leaves()

    dims_and_sizes = self.root._get_dims()
    self.order = np.argsort([dim for dim, size in dims_and_sizes])
    self.shape = tuple([size for dim, size in sorted(dims_and_sizes)])

  def __getitem__(self, slices):
    assert len(slices) == self.d, 'Dimensionality mismatch.'
    eval = self.root[slices].squeeze(-1)
    # reorder dims
    return np.transpose(eval, self.order)
  
  def to_full(self):
    slices = [slice(None) for _ in range(self.d)]
    return self[slices]
  
  def __str__(self,):
    return self.root._print(indent=0)

  # def shape(self,):
  #   assert self.tensor_shape is not None, "Run setup() fist."
  #   return self.tensor_shape

class HTTreeNode:
  '''
  Node for the hierachical Tucker tree.
  '''
  def __init__(self, children=None, r : int = 1, n: int = None, dim :int = None, core_init=None):
    self.parent = None
    self.children = children
    self.r = r

    assert children is None or len(children) == 2, "Only binary trees allowed."

    if n is not None and dim is not None:
      assert self.isleaf(), "Only leaf nodes can have a dimension."
      assert n >= r, f"Leaf rank can not be larger than dim size ({r} > {n})." 
      self.dim = dim
      self.n = n
      if core_init is None:
        self.core = np.zeros((n,r))
      else:
        assert core_init.shape == (n,r), \
          f"Core with shape mismatch: {core_init.shape} != {(n,r)}."
        self.core = core_init
      
    else:
      assert not self.isleaf(), f"Leaf nodes must have a dimension."
      if core_init is None:
        self.core = np.zeros((self.children[0].r, self.children[1].r, self.r,))
      else:
        assert core_init.shape == (self.children[0].r, self.children[1].r, self.r,), \
          f"Core init shape mismatch: {core_init.shape} != {(self.children[0].r, self.children[1].r, self.r,)}."
        self.core = core_init

    if not self.isleaf():
      for child in children:
        child.parent = self

  def isleaf(self):
    return self.children is None

  def isroot(self):
    return self.parent is None
  
  def __getitem__(self, slices):
    if self.isleaf():
      res = np.atleast_2d(self.core[slices[self.dim]])
    else:
      branch_evals = [child[slices] for child in self.children]
      shapes = [eval.shape[:-1] for eval in branch_evals]
      branch_evals = [eval.reshape(-1, eval.shape[-1]) for eval in branch_evals]
      res = np.einsum('ai,bj,ijk->abk', branch_evals[0], branch_evals[1], self.core)
      res = res.reshape(shapes[0] + shapes[1] + (-1,))

    return np.ascontiguousarray(res)
    
  def _count_leaves(self):
    if self.isleaf(): return 1
    else:
      return sum([child._count_leaves() for child in self.children])
    
  def _print(self, indent):
    if self.isleaf():
      return indent * ' ' + f'Leaf node for dim {self.dim} of size {self.n} \n'
    else:
      str = indent * ' ' + f'Interior node, ranks {(self.children[0].r, self.children[1].r, self.r,)} \n'
      for child in self.children:
        str += child._print(indent + 2)
      return str
    
  def _get_dims(self):
    if self.isleaf():
      return [(self.dim, self.n)]
    else:
      dims_and_sizes = []
      for child in self.children:
        dims_and_sizes += child._get_dims()
      return dims_and_sizes


class HTCrossTree(HTTree):
  '''
  Hierarchical Tucker tensor specialication with cross approximation methods.
  '''

  def init_randn(self):
    self.root._init_randn()

  def run_cross(self, eval_f, iterations=10, eps=1e-3):
    self.root._init_indexset()
    ind_data = (np.zeros((1,0)), [])
    even = False
    for iter in range(iterations):
      self.root._run_cross(eval_f, ind_data, kickrank=1, eps=eps, dir=even)
      # even = not even
    for iter in range(iterations):
      self.root._run_cross(eval_f, ind_data, kickrank=0, eps=eps, dir=even)
    #   # even = not even


class HTCrossTreeNode(HTTreeNode):
  def _setup(self,):
    pass

  def _init_randn(self):
    self.core = np.random.randn(*self.core.shape)
    if not self.isleaf():
      for child in self.children:
        child._init_randn()

  def _init_indexset(self):
    if self.isleaf():
      self.indexset = np.random.choice(np.arange(self.n), replace=False, size=(self.r, 1))
      self.indexset_dims = [self.dim]
    else:
      for child in self.children:
        child._init_indexset()
      
      if not self.isroot():
        indexset_dims = np.concatenate([child.indexset_dims for child in self.children])
        order = np.argsort(indexset_dims)
        self.indexset_dims = indexset_dims[order]
        indexset = np.hstack(
          (np.repeat(self.children[0].indexset, self.children[1].r, axis=0),
          np.tile(self.children[1].indexset, (self.children[0].r, 1)))
          )
        
        indexset = indexset[:, order]
        ind = np.random.choice(
          np.arange(self.children[0].r * self.children[1].r), replace=False, size=(self.r)
          )
        self.indexset = indexset[ind]

  def _run_cross(self, eval_f, ind_data, kickrank, eps, dir):
    # kickrank = 1
    rf = 2
    # eps = 1e-1
    # nested maxvol indices comming from the parent
    indexset_p, indexset_dims_p = ind_data

    if self.isleaf():
      # assemble index set (in C-order for n x r matrix)
      indexset_dims = np.concatenate((indexset_dims_p, np.array([self.dim], dtype=int)))
      order = np.argsort(indexset_dims)
      indexset = np.hstack(
        (np.tile(indexset_p, (self.n,1)), np.repeat(np.arange(self.n), self.r).reshape(-1,1))
      )
      indexset = indexset[:, order]
      # eval tensor via black box function
      eval = eval_f(indexset)
      eval = eval.reshape(self.n, self.r, order='C')
      
      # q,r = np.linalg.qr(eval)
      # q,r = localcross(eval, tol=1e-5)
      # self.r = q.shape[1]
      # ind, _ = maxvol(q)
      # qmax = q[ind]
      # self.core = np.linalg.solve(qmax.T, q.T).T

      u,s,v = svd_cut(eval, tol=eps)
      r = np.diag(s) @ v
      ind, C = rect_maxvol(u, tol=1.1, maxK=u.shape[0] + kickrank + rf, min_add_K=kickrank)
      self.r = len(ind)
      qmax = u[ind]
      
      self.core = C

      self.indexset_dims = np.array([self.dim], dtype=int)
      self.indexset = ind.reshape(-1,1)
      return qmax @ r
    
    else:
      if dir:
        child_pairings = [(1, 0), (0, 1),]
      else:
        child_pairings = [(0, 1), (1, 0),]
      for child_i, other_child_i in child_pairings:
        child = self.children[child_i]
        other_child = self.children[other_child_i]
        # move interface axis to front
        core = np.swapaxes(self.core, 0, child_i)
        core = np.ascontiguousarray(core)
        core = core.reshape(child.r, -1)

        # q,r = np.linalg.qr(core.T)
        # q,r = localcross(core.T, tol=1e-5)
        # child.r = q.shape[1]
        # ind, _ = maxvol(q)
        # qmax = q[ind]
        # core = np.linalg.solve(qmax.T, q.T)

        # truncate and orth
        u,s,v = svd_cut(core.T, tol=eps)
        r = np.diag(s) @ v
        ind, C = rect_maxvol(u, tol=1.1, maxK=u.shape[0] + kickrank + rf, min_add_K=kickrank)
        child.r = len(ind)
        qmax = u[ind]
        core = C.T
        # print(core.flags['C_CONTIGUOUS'])

        child.core = np.einsum('...i, ij, jk->...k', child.core, r.T, qmax.T)

        # update index set
        indexset_dims = np.concatenate([other_child.indexset_dims, indexset_dims_p])
        order = np.argsort(indexset_dims)
        indexset_dims = indexset_dims[order]
        indexset = np.hstack(
          (np.repeat(other_child.indexset, self.r, axis=0),
          np.tile(indexset_p, (other_child.r, 1)))
          )
        indexset = indexset[:, order]
        indexset = indexset[ind]

        # move to child
        factor = child._run_cross(eval_f, (indexset, indexset_dims), kickrank, eps, dir)

        # update core
        core = (factor @ core)
        core = np.ascontiguousarray(core)
        core = core.reshape(child.r, other_child.r, self.r)
        self.core = np.swapaxes(core, 0, child_i)
        # self.core = np.ascontiguousarray(self.core)

      # orth towards parent
      if not self.isroot():
        core = np.ascontiguousarray(self.core)
        core = core.reshape(-1, self.r)

        # q,r = np.linalg.qr(core)
        # q, r = localcross(core, tol=1e-5)
        # self.r = q.shape[1]
        # ind, _ = maxvol(q)
        # qmax = q[ind]
        # core = np.linalg.solve(qmax.T, q.T).T

        u,s,v = svd_cut(core, tol=eps)
        r = np.diag(s) @ v
        ind, C = rect_maxvol(u, tol=1.1, maxK=u.shape[0] + kickrank + rf, min_add_K=kickrank)
        self.r = len(ind)
        qmax = u[ind]

        core = np.ascontiguousarray(C)
        self.core = core.reshape(self.children[0].r, self.children[1].r, self.r)

        # update index set towards parent
        indexset_dims = np.concatenate([child.indexset_dims for child in self.children])
        order = np.argsort(indexset_dims)
        self.indexset_dims = indexset_dims[order]
        indexset = np.hstack(
          (np.repeat(self.children[0].indexset, self.children[1].r, axis=0),
          np.tile(self.children[1].indexset, (self.children[0].r, 1)))
          )
        indexset = indexset[:, order]
        self.indexset = indexset[ind]

        return qmax @ r
      
      # truncate only for root
      else:
        core = self.core.reshape(-1, self.children[0].r)
        u,s,v = svd_cut(core, tol=eps)
        core = u @ np.diag(s) @ v
        self.core = core.reshape(self.children[0].r, self.children[1].r, self.r)

