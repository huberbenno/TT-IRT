"""
numpy and torch implementations behind a common interface, so that tests can be
parametrized over the backend (see the `backend` fixture in conftest.py).
"""
import numpy as np
import torch

from tree_tensor import tree_tensor as tt_numpy, tree_tensor_torch as tt_torch
from tree_cross import tree_cross as tc_numpy, tree_cross_torch as tc_torch
from tree_als_cross import tree_als_cross as als_numpy, tree_als_cross_torch as als_torch


class NumpyProblem:
  """
  assem_solve_fun for the numpy TreeALSCross, wrapping one that works on torch tensors.
  """
  def __init__(self, problem):
    self.problem = problem

  @staticmethod
  def _to_torch(arrays):
    return [torch.as_tensor(np.ascontiguousarray(a)) for a in arrays]

  def matrix(self, cores):
    return [[m.numpy(force=True) for m in mats] for mats in self.problem.matrix(self._to_torch(cores))]

  def rhs(self, cores):
    return [[f.numpy(force=True) for f in fs] for fs in self.problem.rhs(self._to_torch(cores))]

  def solve(self, coeff_A, coeff_b):
    return [u.numpy(force=True) for u in self.problem.solve(self._to_torch(coeff_A), self._to_torch(coeff_b))]


class NumpyBackend:
  TT = tt_numpy.TreeBasedTensor
  TC = tc_numpy.TreeCross
  ALS = als_numpy.TreeALSCross
  float32, float64 = np.float32, np.float64

  @staticmethod
  def randn(tree, shape, rank, seed, dtype=np.float64):
    return NumpyBackend.TT.randn(tree, shape, rank, seed=seed, dtype=dtype)

  @staticmethod
  def from_numpy(a):
    return a

  @staticmethod
  def from_torch_tensor(x):
    """ numpy TreeBasedTensor from a torch TreeBasedTensor. """
    return NumpyBackend.TT([c.numpy(force=True) for c in x.cores], x.tree)

  @staticmethod
  def wrap_problem(problem):
    """ assem_solve_fun for the backend from one working on torch tensors. """
    return NumpyProblem(problem)

class TorchBackend:
  TT = tt_torch.TreeBasedTensor
  TC = tc_torch.TreeCross
  ALS = als_torch.TreeALSCross
  float32, float64 = torch.float32, torch.float64

  @staticmethod
  def randn(tree, shape, rank, seed, dtype=torch.float64, device=None):
    return TorchBackend.TT.randn(tree, shape, rank, seed=seed, dtype=dtype, device=device)

  @staticmethod
  def from_numpy(a):
    return torch.from_numpy(a)

  @staticmethod
  def from_torch_tensor(x):
    return x

  @staticmethod
  def wrap_problem(problem):
    return problem
