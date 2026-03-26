from tree import Tree, NodeIndexedList
import numpy as np
from maxvolpy.maxvol import svd_cut, rect_maxvol
import copy

###### Borrows from tensap, keep in mind if publishing! #####


class TreeBasedTensor:
  def __init__(self, cores, tree: Tree = None):
    if (tree is None) and isinstance(cores, TreeBasedTensor):
      # Create a copy
      self.tree = copy.deepcopy(cores.tree)
      self.cores = copy.deepcopy(cores.cores)
      self.shape = cores.shape
    elif isinstance(cores, (list, np.ndarray)) and isinstance(
        tree, Tree):
      assert tree.n_nodes == len(cores), f'Number of tree nodes must match number of cores ({tree.n_nodes} != {len(cores)}).'
      self.tree = copy.deepcopy(tree)
      self.cores = NodeIndexedList([np.array(x) for x in cores])
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
      return val.squeeze(-1).transpose(np.argsort(order))
    else:
      return val.squeeze(-1)

  def _getitem_subtree(self, node, slices):
    if node.isleaf:
      return np.atleast_2d(self.cores[node][slices[node.dim]]), [node.dim]
    else:
      tmp = self.cores[node]
      n_c = node.n_children + 1
      order = []
      for i, child in enumerate(node.children):
        val_c, order_c = self._getitem_subtree(child, slices)
        order = order_c + order
        tmp = np.tensordot(val_c, tmp, axes=(-1, -n_c + i))
      return tmp, order

  def numpy(self) -> np.ndarray:
    """
    Get a dense representation of the tensor.
    """
    slices = [slice(None) for _ in range(self.ndim)]
    return self[slices]

  @staticmethod
  def randn(tree: Tree, shape, rank: int, seed=None) -> TreeBasedTensor:
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

    rng = np.random.default_rng(seed)
    cores = []
    for node in tree.node_list:
      if node.isleaf:
        cores += [rng.standard_normal((shape[node.dim], rank))]
      elif node.isroot:
        cores += [rng.standard_normal(node.n_children * (rank,) + (1,))]
      else:
        cores += [rng.standard_normal(node.n_children * (rank,) + (rank,))]

    return TreeBasedTensor(cores, tree)
  
  @staticmethod
  def ones(tree: Tree, shape) -> TreeBasedTensor:
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
        cores += [np.ones((shape[node.dim], 1))]
      elif node.isroot:
        cores += [np.ones(node.n_children * (1,) + (1,))]
      else:
        cores += [np.ones(node.n_children * (1,) + (1,))]

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
        new_core = np.einsum('ai,aj->aij', core1, core2)
        self.cores[node] = new_core.reshape(self.shape[node.dim], -1)

      else:
        perm_interleave = np.empty(2 * core1.ndim, dtype=int)
        perm_interleave[::2] = np.arange(core1.ndim)
        perm_interleave[1::2] = np.arange(core1.ndim, 2* core1.ndim)
        new_shape = np.array(core1.shape) * np.array(core2.shape)
        self.cores[node] = np.tensordot(core1, core2, axes=0).transpose(perm_interleave).reshape(new_shape)

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
        self.cores[node] = np.concatenate((core1, core2), axis=-1)

      else:
        for child in node.children:
          worker(child)

        new_shape = np.array(core1.shape) + np.array(core2.shape)
        if node.isroot: new_shape[-1] = 1
        new_core = np.zeros(new_shape)
        slices1 = tuple(slice(d) for d in core1.shape)
        new_core[*slices1] = core1
        slices2 = tuple(slice(-d, None) for d in core2.shape)
        new_core[*slices2] = core2
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
        core = np.tensordot(core1, core2, axes=0).transpose(perm_interleave).reshape(new_shape)

        for child in node.children:
          partial = worker(child)
          core = np.tensordot(partial, core, axes=(0,0))

        return core

    return worker(self.tree.root)

  def norm(self):
    """
    Compute the Frobenius norm of the tensor.
    """
    return np.sqrt(self.dot(self))[0]

  def round(self, tol=1e-3) -> TreeBasedTensor:
    """
    Truncate tensor ranks to specified tolerance using SVD.

    TODO relative tol
    """
    r = self._orth_subtree(self.tree.root)
    self.cores[self.tree.root] *= r

    def worker_trunc(node):
      core = self.cores[node]
      if node.isleaf:
        u,s,v = svd_cut(core, tol=tol)
        self.cores[node] = u @ np.diag(s) @ v

      else:
        for i, child in enumerate(node.children):
          r_c = core.shape[i]
          core = np.swapaxes(core, -1, i)
          old_shape = core.shape[:-1]
          core = core.reshape(-1, r_c)
          u,s,v = svd_cut(core, tol=tol)
          core = u
          self.cores[child] = np.tensordot(self.cores[child], np.diag(s) @ v, axes=(-1,-1))
          worker_trunc(child)
          core = core.reshape(old_shape + (-1,))
          core = np.swapaxes(core, -1, i)

        self.cores[node] = core

    worker_trunc(self.tree.root)

    return self
  
  def _orth_subtree(self, node):
    core = self.cores[node]
    if node.isleaf:
      q,r = np.linalg.qr(core)
      self.cores[node] = q
      return r
    else:
      old_shape = core.shape
      for i, child in enumerate(node.children):
        r = self._orth_subtree(child)
        core = np.tensordot(core, r, axes=(i,-1))
        core = np.moveaxis(core, -1, i)

      core = core.reshape(-1, core.shape[-1])
      q,r = np.linalg.qr(core)
      self.cores[node] = q.reshape(old_shape)
      return r
    
  def _orth_subtree_maxvol(self, node):
    indexset_list = NodeIndexedList(self.tree.n_nodes * [None])
    indexset_dims_list = NodeIndexedList(self.tree.n_nodes * [None])
    maxvol_ind_list = NodeIndexedList(self.tree.n_nodes * [None])

    # recursive worker function
    def worker(node):
      core = self.cores[node]
      if node.isleaf:
        q, r = np.linalg.qr(core)
        ind, C = rect_maxvol(q, maxK=core.shape[-1])
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
          core = np.tensordot(core, r, axes=(i,-1))
          core = np.moveaxis(core, -1, i)
          c_dims += [indexset_dims_list[child]]

        if node.isroot:
          self.cores[node] = core
          return None
        else:
          dims = np.concatenate(c_dims)
          # build combined index set
          ind_grid = np.meshgrid(
            *[np.arange(indexset_list[c].shape[0]) for c in node.children]
          )
          indexset = np.concatenate(
            [*[indexset_list[c][grid.flatten()] for c, grid in zip(node.children, ind_grid)]],
            axis = -1
          )

          # find maxvol indices
          old_shape = core.shape
          q, r = np.linalg.qr(core.reshape(-1, core.shape[-1]))
          ind, C = rect_maxvol(q, maxK=core.shape[-1])
          qmax = q[ind]
          
          indexset_list[node] = indexset[ind]
          indexset_dims_list[node] = dims
          maxvol_ind_list[node] = ind

          self.cores[node] = C.reshape(old_shape[:-1] + (-1,))
          return qmax @ r

    r = worker(node)

    return r, indexset_list, indexset_dims_list


  def print(self):
    """
    Print a representation of the tree tensor.
    """
    print(f"Tree tensor with shape {self.shape}")
    print(self._print(self.tree.root),end='')

  def _print(self, node, prefix='', last=False):
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