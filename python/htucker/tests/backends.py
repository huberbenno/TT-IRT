"""
numpy and torch implementations behind a common interface, so that tests can be
parametrized over the backend (see the `backend` fixture in conftest.py).
"""
import numpy as np
import torch

from tree_tensor import tree_tensor as tt_numpy, tree_tensor_torch as tt_torch
from tree_cross import tree_cross as tc_numpy, tree_cross_torch as tc_torch


class NumpyBackend:
  TT = tt_numpy.TreeBasedTensor
  TC = tc_numpy.TreeCross
  float32, float64 = np.float32, np.float64

  @staticmethod
  def randn(tree, shape, rank, seed, dtype=np.float64):
    return NumpyBackend.TT.randn(tree, shape, rank, seed=seed, dtype=dtype)

  @staticmethod
  def from_numpy(a):
    return a

class TorchBackend:
  TT = tt_torch.TreeBasedTensor
  TC = tc_torch.TreeCross
  float32, float64 = torch.float32, torch.float64

  @staticmethod
  def randn(tree, shape, rank, seed, dtype=torch.float64, device=None):
    return TorchBackend.TT.randn(tree, shape, rank, seed=seed, dtype=dtype, device=device)

  @staticmethod
  def from_numpy(a):
    return torch.from_numpy(a)
