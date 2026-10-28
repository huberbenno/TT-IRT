import os
import json
import numpy as np
import torch
from dataclasses import asdict

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if np.issubdtype(obj, np.integer):
      return int(obj)
    return super().default(obj)

def save_stats(args, stats, subdir, prefix='stats'):
  """ Save args and stats to data/<subdir>/<prefix>_<hash of args>.json, the path is also stored in stats['file']. """
  fn = f'data/{subdir}/{prefix}_{hash(args):X}.json'
  os.makedirs(os.path.dirname(fn), exist_ok=True)
  stats['file'] = fn
  with open(fn, mode='w') as file:
    json.dump({'args': asdict(args), 'stats':stats}, file, cls=NumpyEncoder)
  return fn

def RMS(x, **kwargs):
  return np.sqrt(np.mean(np.square(x), **kwargs))

def coeff_mc_errors(tensor, cfun, rng, N_mc):
  """
  Monte Carlo estimate of the coefficient approximation error.

  Parameters
  ----------
  tensor: TreeBasedTensor (torch)
    Approximation of the coefficient, dimension 0 is the physical one.
  cfun: callable
    Coefficient at a batch of indices (N, ndim), as a torch tensor.
  rng: np.random.Generator
    Draws the parameter indices, one per sample.
  N_mc: int
    Number of random parameter indices.

  Returns
  -------
  errs: np.ndarray
    (N_mc,) RMS errors over the physical dimension at the random parameter indices.
  """
  # One fibre over the physical dimension per sample. This is faster than
  # tensor.entries on all N_mc * Nx indices for larger ranks, since slicing only
  # carries the physical dimension along the path from its leaf to the root.
  Nx, param_shape = tensor.shape[0], tensor.shape[1:]
  errs = np.zeros(N_mc)
  with torch.no_grad():
    for i in range(N_mc):
      y = rng.integers(0, param_shape)
      C_eval = torch.ravel(tensor[:, *y])
      X = np.hstack([np.arange(Nx).reshape((-1, 1)), np.tile(y, [Nx, 1])])
      errs[i] = RMS((C_eval - cfun(X).to(device=C_eval.device)).numpy(force=True))
  return errs
