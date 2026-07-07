import numpy as np

from typing import Callable
from tree_tensor import TreeBasedTensor, TreeNode
from tree import NodeIndexedList
from maxvolpy.maxvol import rect_maxvol, svd_cut

def RMS(x, **kwargs):
  return np.sqrt(np.mean(np.square(x), **kwargs))

class TreeCross:
  def __init__(self, tensor: TreeBasedTensor):
    self.tensor = tensor
    self.indexset_list = None
    self.indexset_dims_list = None
    self.n_eval = 0

  def run(
      self, 
      eval_f: Callable[[np.array]], 
      n_iter: int=1, 
      eps: float=0., 
      kickrank: int=2, 
      rf=2, 
      force_init: bool=True,
      resample: bool=False,
      verbose: bool|int=False
      ) -> None:
    """
    Run the cross approximation.

    Parameters
    ----------
    eval_f: callable
      Black-box function for evaluating the target at arbitrary indices.
      Must take (N, ndim) arrays of batched indices.
    n_iter: int, optional
      Number of iterations to run. Default is 1.
    eps: float, optional
      Tolerance for rank truncation. Default is 0 (no truncation).
    kickrank: int, optional
      Number of ranks to add in each step.
    rf: int, optional
      Tuning parameter.
    force_init: bool, optional
      Force indexset reinitialization. 
    resample: bool, optional
      Resample interior nodes after subtree recursion.
    verbose: bool or int, optional
      Print diagnostics if True.
    """
    if self.indexset_list is None or force_init:
      self._init()

    self.max_dx = 0
    eps_extra = 1e-1

    def worker(node: TreeNode, indexset_p: np.ndarray, indexset_dims_p: np.ndarray):
      if node.isleaf:
        n, r = self.tensor.shape[node.dim], self.tensor.cores[node].shape[-1]
        dims = np.concatenate((indexset_dims_p, np.array([node.dim], dtype=int)))
        order = np.argsort(dims)
        indexset = np.hstack(
          (np.tile(indexset_p, (n,1)), np.repeat(np.arange(n), r).reshape(-1,1))
        )
        eval = eval_f(indexset[:, order])
        self.n_eval += eval.shape[0]
        eval = eval.reshape(n, r)

        # compute local residual
        res = RMS(eval - self.tensor.cores[node].reshape(n,r))
        self.max_dx = max(res, self.max_dx)
        if verbose > 1: print(f'node {node.id}, res = {res:.2e}')

        u,s,v = svd_cut(eval, tol=eps_extra*eps/np.sqrt(self.tensor.ndim), norm='fro')
        r = np.diag(s) @ v
        ind, C = rect_maxvol(u, tol=1.1, maxK=u.shape[1] + kickrank + rf, min_add_K=kickrank)
        qmax = u[ind]
        self.tensor.cores[node] = C

        self.indexset_list[node] = ind.reshape(-1,1)
        return qmax @ r
      
      else:
        if resample:
          # sample core at indexset
          indexset = self._assemble_indexset(node, indexset_p, indexset_dims_p)
          eval = eval_f(indexset)
          self.n_eval += eval.shape[0]
          core = eval.reshape(self.tensor.cores[node].shape)

          # compute local residual
          res = RMS(core - self.tensor.cores[node])
          self.max_dx = max(res, self.max_dx)
          if verbose > 1: print(f'node {node.id}, res = {res:.2e}')

          self.tensor.cores[node] = core

        for i, child in enumerate(node.children):
          core = self.tensor.cores[node]

          # orth toward child
          core = np.moveaxis(core, i, -1)
          old_shape = core.shape[:-1]
          core = core.reshape(-1, core.shape[-1])
          u,s,v = svd_cut(core, tol=eps_extra*eps/np.sqrt(self.tensor.ndim), norm='fro')
          # u,s,v = np.linalg.svd(core, full_matrices=False)
          r = np.diag(s) @ v
          ind, C = rect_maxvol(u, tol=1.1, maxK=u.shape[1] + kickrank + rf, min_add_K=kickrank)
          qmax = u[ind]
          core = C
          # push non orth factor to child
          self.tensor.cores[child] = np.tensordot(self.tensor.cores[child], qmax @ r, axes=(-1,-1))

          # update index set
          indexset_sizes = [np.arange(self.indexset_list[c].shape[0]) for c in child.siblings] + [np.arange(indexset_p.shape[0])]
          ind_grid = np.meshgrid(*indexset_sizes, indexing='ij')
          indexsets = [self.indexset_list[c] for c in child.siblings] + [indexset_p]
          indexset = np.concatenate(
            [ind[grid.flatten()] for ind, grid in zip(indexsets, ind_grid)],
            axis = -1
          )
          indexset = indexset[ind]
          indexset_dims = np.concatenate([self.indexset_dims_list[c] for c in child.siblings] + [indexset_dims_p])

          # go to child
          factor = worker(child, indexset, indexset_dims)
          # absord non orth factor from child
          core = core.reshape(old_shape + (-1,))
          # TODO can skip this for last child
          core = np.tensordot(core, factor, axes=(-1,-1))
          self.tensor.cores[node] = np.moveaxis(core, -1, i)

        # orth towards parent
        if not node.isroot:
          #sample core at indexset
          indexset = self._assemble_indexset(node, indexset_p, indexset_dims_p)
          eval = eval_f(indexset)
          self.n_eval += eval.shape[0]
          core = eval.reshape(self.tensor.cores[node].shape)

          # compute local residual
          res = RMS(core - self.tensor.cores[node])
          self.max_dx = max(res, self.max_dx)
          if verbose > 1: print(f'node {node.id}, res = {res:.2e}')

          old_shape = core.shape[:-1]
          core = core.reshape(-1, core.shape[-1])

          u,s,v = svd_cut(core, tol=eps_extra*eps/np.sqrt(self.tensor.ndim), norm='fro')
          # print(f'{s[0]/s[-1]:.2e}')
          # u,s,v = np.linalg.svd(core, full_matrices=False)
          r = np.diag(s) @ v
          ind, C = rect_maxvol(u, tol=1.1, maxK=u.shape[1] + kickrank + rf, min_add_K=kickrank)
          qmax = u[ind]

          core = C.reshape(old_shape + (-1,))
          self.tensor.cores[node] = core

          # update index set towards parent
          ind_grid = np.meshgrid(
            *[np.arange(self.indexset_list[c].shape[0]) for c in node.children],
            indexing='ij'
          )
          indexset = np.concatenate(
            [*[self.indexset_list[c][grid.flatten()] for c, grid in zip(node.children, ind_grid)]],
            axis = -1
          )
          self.indexset_list[node] = indexset[ind]

          return qmax @ r

    
    for iteration in range(n_iter):
      self.max_dx = 0
      worker(self.tensor.tree.root, np.zeros((1,0)), [])

      if verbose > 0:
        print(
          f'swp: {str(iteration+1).rjust(len(str(n_iter)))}/{n_iter}',
          f'f_evals={self.n_eval}',
          f'self.max_dx={self.max_dx:.2e}'
          )
      if self.max_dx < eps:
        break

  def _init(self):
    _, self.indexset_list, self.indexset_dims_list, _ = self.tensor._orth_subtree_maxvol(self.tensor.tree.root)

  def _assemble_indexset(self, node: TreeNode, indexset_p: np.ndarray, indexset_dims_p: np.ndarray):
    indexset_sizes = [np.arange(self.indexset_list[c].shape[0]) for c in node.children] + [np.arange(indexset_p.shape[0])]
    ind_grid = np.meshgrid(*indexset_sizes, indexing='ij')
    indexsets = [self.indexset_list[c] for c in node.children] + [indexset_p]
    indexset = np.concatenate(
      [ind[grid.flatten()] for ind, grid in zip(indexsets, ind_grid)],
      axis = -1
    )
    indexset_dims = np.concatenate([self.indexset_dims_list[c] for c in node.children] + [indexset_dims_p])
    order = np.argsort(indexset_dims)
    
    return indexset[:, order]