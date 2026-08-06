from tree import Tree, TreeNode, NodeIndexedList
import numpy as np
import torch
from maxvolpy.maxvol import rect_maxvol
import copy

###### Borrows from tensap, keep in mind if publishing! #####

# torchified version of maxvolpy.maxvol.svd_cut 
def svd_cut_torch(A, tol, alpha=0., norm=2):
    """
    Computes SVD and cuts low singular values.
    
    Computes singular values decomposition of matrix `A`, adds
    regularizing parameter `alpha` to each singular value and returns
    only largest singular values and vectors with relative tolerance
    `tol`.

    Parameters
    ----------
    A: torch.tensor
        Real or complex matrix or matrix-like object.
    tol: float
        Tolerance of cutting singular values operation.
    alpha: float, optional
        Regularizing parameter.
    norm: {2, 'fro'}, optional
        Defines norm, that is chosen when cutting singular values.

    Returns
    -------
    U: torch.tensor
        Left singular vectors, corresponding to largest singular values.
    S: torch.tensor
        Largest singular values.
    V: torch.tensor
        Right singular vectors, corresponding to largest singular values.
    """
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    S_reg = S+alpha
    S1 = S_reg.numpy(force=True)[::-1]
    if norm == 2:
        rank = S1.shape[0]-np.searchsorted(S1, tol*S1[-1], side='left')
    elif norm == 'fro':
        S1 = np.cumsum(np.square(S1))
        rank = S1.shape[0]-np.searchsorted(S1, S1[-1] * tol**2, side='left')
    else:
        raise ValueError("Invalid parameter norm value")
    return U[:,:rank], S_reg[:rank], V[:rank]


class TreeBasedTensor:
  def __init__(self, cores, tree: Tree = None, dtype=torch.float64, device=None):
    self.dtype = dtype
    self.device = device
    if (tree is None) and isinstance(cores, TreeBasedTensor):
      # Create a copy
      self.tree = copy.deepcopy(cores.tree)
      self.cores = copy.deepcopy(cores.cores)
      self.shape = cores.shape

    elif isinstance(cores, (list, torch.tensor)) and isinstance(
        tree, Tree):
      assert tree.n_nodes == len(cores), f'Number of tree nodes must match number of cores ({tree.n_nodes} != {len(cores)}).'
      if dtype is None: self.dtype = cores[0].dtype
      if device is None: self.device = cores[0].device
      self.tree = copy.deepcopy(tree)
      self.cores = NodeIndexedList([x.detach().clone().to(device=self.device, dtype=self.dtype) for x in cores])
      self.shape = tuple(cores[tree.dim2id(i)].shape[0] for i in range(tree.order))

    elif isinstance(cores, (list, np.ndarray)) and isinstance(
        tree, Tree):
      assert tree.n_nodes == len(cores), f'Number of tree nodes must match number of cores ({tree.n_nodes} != {len(cores)}).'
      self.tree = copy.deepcopy(tree)
      self.cores = NodeIndexedList([torch.from_numpy(x).to(device=self.device, dtype=self.dtype) for x in cores])
      self.shape = tuple(cores[tree.dim2id(i)].shape[0] for i in range(tree.order))

    else:
      raise NotImplementedError(
        "Constructor not implemented for the " "provided arguments."
      )

  @property
  def ndim(self):
    """
    Number of dimensions of the tensor. Same as len(shape) and self.tree.order.
    """
    return len(self.shape)

  @property
  def size(self):
    """
    Number of entries of the full tensor.
    """
    return np.prod(self.shape)

  @property
  def sparse_size(self):
    """
    Number of variables used for tensor storage.
    """
    return np.sum([core.size for core in self.cores])

  def __getitem__(self, slices, ordered = True):
    assert self.tree.order == len(slices), \
      f'Tree order must match number of sliced dimensions ({self.tree.order} != {len(slices)}).'

    val, order = self._getitem_subtree(self.tree.root, slices)
    if ordered:
      return val.squeeze(-1).permute(np.argsort(order).tolist())
    else:
      return val.squeeze(-1)

  def _getitem_subtree(self, node, slices):
    if node.isleaf:
      return torch.atleast_2d(self.cores[node][slices[node.dim]]), [node.dim]
    else:
      tmp = self.cores[node]
      n_c = node.n_children + 1
      order = []
      for i, child in enumerate(node.children):
        val_c, order_c = self._getitem_subtree(child, slices)
        order = order_c + order
        tmp = torch.tensordot(val_c, tmp, dims=((-1,), (-n_c + i,)))
        
      return tmp, order

  def numpy(self) -> np.ndarray:
    return self.to_torch().numpy(force=True)

  def to_torch(self) -> torch.tensor:
    slices = [slice(None) for _ in range(self.ndim)]
    return self[slices]

  @staticmethod
  def randn(tree: Tree, shape, rank: int, seed=None, dtype=torch.float64, device=None) -> TreeBasedTensor:
    """
    Build a tensor with cores initizialized from the standard normal distribution.

    Parameters
    ----------
    tree: Tree
      Object defining the dimension tree strucure.
    shape: tuple(int, ...)
      Shape of the tensor.
    rank: int
      Uniform rank for initialization.
    seed: int, optional
      Seed for the rng. Default uses system entropy.
    """
    assert tree.order == len(shape), \
      f'Tree order must match shape ({tree.order} != {len(shape)}).'

    rng = torch.Generator()
    if seed is not None: rng.manual_seed(seed)
    cores = []
    for node in tree.node_list:
      if node.isleaf:
        cores += [torch.randn((shape[node.dim], rank), generator=rng, dtype=dtype)]
      elif node.isroot:
        cores += [torch.randn(node.n_children * (rank,) + (1,), generator=rng, dtype=dtype)]
      else:
        cores += [torch.randn(node.n_children * (rank,) + (rank,), generator=rng, dtype=dtype)]

    return TreeBasedTensor(cores, tree, dtype=dtype, device=device)

  @staticmethod
  def ones(tree: Tree, shape, dtype=torch.float64) -> TreeBasedTensor:
    """
    Build a tensor (of rank 1) filled with ones .

    Parameters
    ----------
    tree: Tree
      Object defining the dimension tree strucure.
    shape: tuple(int, ...)
      Shape of the tensor.
    """
    assert tree.order == len(shape), \
    f'Tree order must match shape ({tree.order} != {len(shape)}).'

    cores = []
    for node in tree.node_list:
      if node.isleaf:
        cores += [torch.ones((shape[node.dim], 1), dtype=dtype)]
      elif node.isroot:
        cores += [torch.ones(node.n_children * (1,) + (1,), dtype=dtype)]
      else:
        cores += [torch.ones(node.n_children * (1,) + (1,), dtype=dtype)]

    return TreeBasedTensor(cores, tree)

  def copy(self):
    return TreeBasedTensor(self)
  
  def __mul__(x, y) -> TreeBasedTensor:
    res_tensor = TreeBasedTensor(x)
    res_tensor *= y
    return res_tensor

  def __rmul__(x, y) -> TreeBasedTensor:
    return x * y

  def __imul__(self, y):
    try:
      # multiplication with a scalar
      scalar = float(y)
      self.cores[self.tree.root] *= scalar
      return self
    except:
      pass

    if not isinstance(y, TreeBasedTensor):
      raise TypeError(f'y must be scalar or TreeBasedTensor, not {type(y)}')

    assert self.tree == y.tree, 'Trees not compatible.'
    assert self.shape == y.shape, f'Shape mismatch: {self.shape} != {y.shape}'

    def worker(node):
      core1 = self.cores[node]
      core2 = y.cores[node]
      if node.isleaf:
        new_core = torch.einsum('ai,aj->aij', core1, core2)
        self.cores[node] = new_core.reshape(self.shape[node.dim], -1)

      else:
        perm_interleave = np.empty(2 * core1.ndim, dtype=int)
        perm_interleave[::2] = np.arange(core1.ndim)
        perm_interleave[1::2] = np.arange(core1.ndim, 2* core1.ndim)
        new_shape = np.array(core1.shape) * np.array(core2.shape)
        self.cores[node] = torch.tensordot(core1, core2, dims=0).transpose(perm_interleave).reshape(new_shape)

        for child in node.children:
          worker(child)

    worker(self.tree.root)
    return self

  def __add__(x,y) -> TreeBasedTensor:
    res_tensor = TreeBasedTensor(x)
    res_tensor += y
    return res_tensor

  def __iadd__(self, y):
    if not isinstance(y, TreeBasedTensor):
      raise TypeError(f'y must be scalar or TreeBasedTensor, not {type(y)}')

    assert self.tree == y.tree, 'Trees not compatible.'
    assert self.shape == y.shape, f'Shape mismatch: {self.shape} != {y.shape}'

    def worker(node):
      core1 = self.cores[node]
      core2 = y.cores[node]
      if node.isleaf:
        self.cores[node] = torch.concatenate((core1, core2), dim=-1)

      else:
        for child in node.children:
          worker(child)

        new_shape = np.array(core1.shape) + np.array(core2.shape)
        if node.isroot: new_shape[-1] = 1
        new_core = torch.zeros(new_shape, device=self.device, dtype=self.dtype)
        slices1 = tuple(slice(d) for d in core1.shape)
        new_core[*slices1] += core1
        slices2 = tuple(slice(-d, None) for d in core2.shape)
        new_core[*slices2] += core2
        self.cores[node] = new_core

    worker(self.tree.root)
    return self

  def __sub__(x, y) -> TreeBasedTensor:
    res_tensor = TreeBasedTensor(x)
    res_tensor -= y
    return res_tensor

  def __isub__(self, y):
    self += -y
    return self

  def __neg__(self) -> TreeBasedTensor:
    neg = TreeBasedTensor(self)
    neg.cores[neg.tree.root] *= -1
    return neg

  def dot(self, other):
    """
    Compute the dot product with another tensor.

    Parameters
    ----------
    other: TreeTensor
      Must have same tree and shape.
    """
    assert self.tree == other.tree, 'Trees not compatible.'
    assert self.shape == other.shape, f'Shape mismatch: {self.shape} != {other.shape}'

    def worker(node):
      core1 = self.cores[node]
      core2 = other.cores[node]
      if node.isleaf:
        return np.einsum('ai,aj->ij', core1, core2).ravel()

      else:
        perm_interleave = np.empty(2 * core1.ndim, dtype=int)
        perm_interleave[::2] = np.arange(core1.ndim)
        perm_interleave[1::2] = np.arange(core1.ndim, 2* core1.ndim)
        new_shape = np.array(core1.shape) * np.array(core2.shape)
        core = torch.tensordot(core1, core2, dims=0).transpose(perm_interleave).reshape(new_shape)

        for child in node.children:
          partial = worker(child)
          core = torch.tensordot(partial, core, dims=((0,),(0,)))

        return core

    return worker(self.tree.root)

  def norm(self):
    """
    Compute the Frobenius norm of the tensor.
    """
    return torch.sqrt(self.dot(self))[0]

  def round(self, tol:float = 0.) -> TreeBasedTensor:
    """
    Truncate tensor ranks to specified tolerance using SVD.
    """
  
    def worker_trunc(node):
      core = self.cores[node]
      if node.isleaf:
        u,s,v = svd_cut_torch(core, tol=tol/np.sqrt(self.ndim), norm='fro')
        self.cores[node] = u
        return torch.diag(s) @ v
      else:
        for i, child in enumerate(node.children):
          r_c = core.shape[i]
          core = torch.swapaxes(core, -1, i)
          old_shape = core.shape[:-1]
          core = core.reshape(-1, r_c)
          u,s,v = svd_cut_torch(core, tol=tol/np.sqrt(self.ndim), norm='fro')
          core = u
          self.cores[child] = torch.tensordot(self.cores[child], torch.diag(s) @ v, dims=((-1,),(-1,)))
          r = worker_trunc(child)
          core = torch.tensordot(core, r, dims=((-1,),(-1,)))
          core = core.reshape(old_shape + (-1,))
          core = torch.swapaxes(core, -1, i)
          self.cores[node] = core

        if not node.isroot:
          old_shape = core.shape[:-1]
          core = core.reshape(-1, core.shape[-1])
          u,s,v = svd_cut_torch(core, tol=tol/np.sqrt(self.ndim), norm='fro')
          core = u.reshape(old_shape + (-1,))
          self.cores[node] = core
          return torch.diag(s) @ v

    worker_trunc(self.tree.root)

    return self

  def _orth_subtree(self, node : TreeNode):
    core = self.cores[node]
    if node.isleaf:
      q,r = torch.linalg.qr(core)
      self.cores[node] = q
      return r
    else:
      old_shape = core.shape
      for i, child in enumerate(node.children):
        r = self._orth_subtree(child)
        core = torch.tensordot(core, r, dims=((i,),(-1,)))
        core = torch.moveaxis(core, -1, i)

      core = core.reshape(-1, core.shape[-1])
      q,r = torch.linalg.qr(core)
      self.cores[node] = q.reshape(old_shape)
      return r

  def _orth_subtree_maxvol(self, node: TreeNode):
    indexset_list = NodeIndexedList(self.tree.n_nodes * [None])
    indexset_dims_list = NodeIndexedList(self.tree.n_nodes * [None])
    maxvol_ind_list = NodeIndexedList(self.tree.n_nodes * [None])

    # recursive worker function
    def worker(node):
      core = self.cores[node]
      if node.isleaf:
        q, r = torch.linalg.qr(core)
        ind, C = rect_maxvol(q.numpy(force=True), maxK=core.shape[-1])
        C = torch.from_numpy(C).to(device=self.device)
        indexset_list[node] = ind.reshape(-1,1)
        indexset_dims_list[node] = (node.dim,)
        maxvol_ind_list[node] = ind
        qmax = q[ind]
        self.cores[node] = C
        return qmax @ r
      else:
        c_dims = []
        for i, child in enumerate(node.children):
          r = worker(child)
          core = torch.tensordot(core, r, dims=((i,), (-1,)))
          core = torch.moveaxis(core, -1, i)
          c_dims += [indexset_dims_list[child]]

        if node.isroot:
          self.cores[node] = core
          return None
        else:
          dims = np.concatenate(c_dims)
          # build combined index set
          ind_grid = np.meshgrid(
            *[np.arange(indexset_list[c].shape[0]) for c in node.children], indexing='ij'
          )
          indexset = np.concatenate(
            [*[indexset_list[c][grid.flatten()] for c, grid in zip(node.children, ind_grid)]],
            axis = -1
          )

          # find maxvol indices
          old_shape = core.shape
          q, r = torch.linalg.qr(core.reshape(-1, core.shape[-1]))
          ind, C = rect_maxvol(q.numpy(force=True), maxK=core.shape[-1])
          C = torch.from_numpy(C).to(device=self.device)
          qmax = q[ind]

          indexset_list[node] = indexset[ind]
          indexset_dims_list[node] = dims
          maxvol_ind_list[node] = ind

          self.cores[node] = C.reshape(old_shape[:-1] + (-1,))
          return qmax @ r

    r = worker(node)

    return r, indexset_list, indexset_dims_list, maxvol_ind_list

  def print(self):
    """
    Print a representation of the tree tensor.
    """
    print(f"Tree tensor with shape {self.shape}")
    print(self._print(self.tree.root),end='')

  def _print(self, node: TreeNode, prefix: str='', last: bool=False):
    if node.isleaf:
      return prefix[3:] + f' + [{node.id}] dim {node.dim} of size {self.shape[node.dim]}\n'
    else:
      if node.isroot:
        str = f'[{node.id}] root ranks={self.cores[node].shape}\n'
      else:
        str = prefix[3:] + f' + [{node.id}] ranks={self.cores[node].shape}\n'

      prefix += '   ' if last else ' | '
      for child in node.children[:-1]:
        str += self._print(child, prefix=prefix)
      str += self._print(node.children[-1], prefix=prefix, last=True)
      return str