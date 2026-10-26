import numpy as np
import torch

from typing import Callable
from tree_tensor.tree_tensor_torch import TreeBasedTensor, svd_cut_torch
from tree.tree import TreeNode
from util.maxvol_torch import rect_maxvol
from tt import set_backend as tt_set_backend
tt_set_backend('torch')

def RMS(x, **kwargs):
  return torch.sqrt(torch.mean(torch.square(x), **kwargs))

class TreeCross:
  def __init__(self, tensor: TreeBasedTensor, seed=None):
    """
    Parameters
    ----------
    tensor: TreeBasedTensor
      Initial guess, updated in place by run.
    seed: int, optional
      Seed for the random index enrichment (random_kick) and the random test
      points (n_test) of run.
    """
    self.tensor = tensor
    self.indexset_list = None
    self.indexset_dims_list = None
    self.n_eval = 0
    self.test_err = None
    self.test_indices = None
    self.test_values = None
    self.rng = np.random.default_rng(seed)

  def run(
      self,
      eval_f: Callable[[np.array]],
      n_iter: int=1,
      eps: float=0.,
      kickrank: int=2,
      force_init: bool=True,
      resample: bool=False,
      verbose: bool|int=False,
      random_kick: int=0,
      n_test: int=0,
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
      Number of ranks to add in each step, chosen greedily by maxvol.
    force_init: bool, optional
      Force indexset reinitialization.
    resample: bool, optional
      Resample interior nodes after subtree recursion.
    verbose: bool or int, optional
      Print diagnostics if True.
    random_kick: int, optional
      Number of randomly drawn rows added to every index selection on top of the
      maxvol rows. Default is 0.
    n_test: int, optional
      If positive, the target is evaluated at n_test new random indices after
      every sweep to test convergence. Default is 0.
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
        eval = eval.to(dtype=self.tensor.dtype, device=self.tensor.device)
        self.n_eval += eval.shape[0]
        eval = eval.reshape(n, r)

        # compute local residual
        res = RMS(eval - self.tensor.cores[node].reshape(n,r))
        self.max_dx = max(res, self.max_dx)
        if verbose > 1: print(f'node {node.id}, res = {res:.2e}')

        u,s,v = svd_cut_torch(eval, tol=eps_extra*eps/np.sqrt(self.tensor.ndim), norm='fro')
        r = torch.diag(s) @ v
        ind, C = self._select_rows(u, kickrank, random_kick)
        C = C.to(device=self.tensor.device, dtype=self.tensor.dtype)
        qmax = u[ind]
        self.tensor.cores[node] = C

        self.indexset_list[node] = ind.reshape(-1,1)
        return qmax @ r
      
      else:
        if resample:
          # sample core at indexset
          indexset = self._assemble_indexset(node, indexset_p, indexset_dims_p)
          eval = eval_f(indexset)
          eval = eval.to(dtype=self.tensor.dtype, device=self.tensor.device)
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
          core = torch.moveaxis(core, i, -1)
          old_shape = core.shape[:-1]
          core = core.reshape(-1, core.shape[-1])
          u,s,v = svd_cut_torch(core, tol=eps_extra*eps/np.sqrt(self.tensor.ndim), norm='fro')
          # u,s,v = np.linalg.svd(core, full_matrices=False)
          r = torch.diag(s) @ v
          ind, C = self._select_rows(u, kickrank, random_kick)
          C = C.to(device=self.tensor.device, dtype=self.tensor.dtype)
          qmax = u[ind]
          core = C
          # push non orth factor to child
          self.tensor.cores[child] = torch.tensordot(self.tensor.cores[child], qmax @ r, dims=((-1,),(-1,)))

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
          core = torch.tensordot(core, factor, dims=((-1,),(-1,)))
          self.tensor.cores[node] = torch.moveaxis(core, -1, i)

        # orth towards parent
        if not node.isroot:
          #sample core at indexset
          indexset = self._assemble_indexset(node, indexset_p, indexset_dims_p)
          eval = eval_f(indexset)
          eval = eval.to(dtype=self.tensor.dtype, device=self.tensor.device)
          self.n_eval += eval.shape[0]
          core = eval.reshape(self.tensor.cores[node].shape)

          # compute local residual
          res = RMS(core - self.tensor.cores[node])
          self.max_dx = max(res, self.max_dx)
          if verbose > 1: print(f'node {node.id}, res = {res:.2e}')

          old_shape = core.shape[:-1]
          core = core.reshape(-1, core.shape[-1])

          u,s,v = svd_cut_torch(core, tol=eps_extra*eps/np.sqrt(self.tensor.ndim), norm='fro')
          # print(f'{s[0]/s[-1]:.2e}')
          # u,s,v = np.linalg.svd(core, full_matrices=False)
          r = torch.diag(s) @ v
          ind, C = self._select_rows(u, kickrank, random_kick)
          C = C.to(device=self.tensor.device, dtype=self.tensor.dtype)
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

        elif not resample and node.n_children > 2:
          # The root is never sampled on the way up. For more than two children, the
          # interpolation from the children's factors keeps a stale basis and the cross
          # stagnates, so sample the root core at the final index sets.
          indexset = self._assemble_indexset(node, indexset_p, indexset_dims_p)
          eval = eval_f(indexset)
          eval = eval.to(dtype=self.tensor.dtype, device=self.tensor.device)
          self.n_eval += eval.shape[0]
          core = eval.reshape(self.tensor.cores[node].shape)

          # compute local residual
          res = RMS(core - self.tensor.cores[node])
          self.max_dx = max(res, self.max_dx)
          if verbose > 1: print(f'node {node.id}, res = {res:.2e}')

          self.tensor.cores[node] = core


    # test points of this run, a new eval_f may be passed in the next one
    self.test_indices = np.zeros((0, self.tensor.ndim), dtype=int)
    self.test_values = torch.zeros(0, dtype=self.tensor.dtype, device=self.tensor.device)
    self.test_err = None

    for iteration in range(n_iter):
      self.max_dx = 0
      worker(self.tensor.tree.root, np.zeros((1,0), dtype=int), [])

      if n_test > 0:
        # independent error estimate at random indices, accumulated over the sweeps
        ind = self.rng.integers(0, self.tensor.shape, size=(n_test, self.tensor.ndim))
        vals = eval_f(ind).to(dtype=self.tensor.dtype, device=self.tensor.device)
        self.n_eval += n_test
        self.test_indices = np.concatenate((self.test_indices, ind))
        self.test_values = torch.cat((self.test_values, vals))
        with torch.no_grad():
          approx = self.tensor.entries(self.test_indices)
        self.test_err = float(RMS(approx - self.test_values) / RMS(self.test_values))

      if verbose > 0:
        print(
          f'swp: {str(iteration+1).rjust(len(str(n_iter)))}/{n_iter}',
          f'f_evals={self.n_eval}',
          f'self.max_dx={self.max_dx:.2e}',
          *([f'test_err={self.test_err:.2e}'] if n_test > 0 else [])
          )
      if self.max_dx < eps and (n_test == 0 or self.test_err < eps):
        break

  def _select_rows(self, u, kickrank, random_kick):
    """
    Rows ind of the orthonormal u and C with u = C u[ind]: the maxvol rows plus
    random_kick randomly drawn ones.
    """
    ind, C = rect_maxvol(u, tol=1.1, maxK=u.shape[1] + kickrank, min_add_K=kickrank, warn_budget=False)
    if random_kick > 0:
      rest = np.setdiff1d(np.arange(u.shape[0]), ind)
      if len(rest) > 0:
        # maxvol only picks rows where the current basis is large, random rows keep
        # exploring the others
        ind = np.concatenate((ind, self.rng.choice(rest, size=min(random_kick, len(rest)), replace=False)))
        C = u @ torch.linalg.pinv(u[ind])
        C[ind] = torch.eye(len(ind), dtype=C.dtype, device=C.device)
    return ind, C

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