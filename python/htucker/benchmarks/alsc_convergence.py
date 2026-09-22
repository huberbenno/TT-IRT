import numpy as np
from dataclasses import dataclass
import pickle

from pde.coeff_function import coeff
from pde.diffusion_1d import Diffusion1D_torch as Diffusion1D

import torch
from tree.tree import Tree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree_cross.tree_cross_torch import TreeCross
from tree.tree_util import balanced_binary_tupletree, linear_tupletree, weighted_binary_tupletree
from tree_als_cross.tree_als_cross_torch import TreeALSCross

@dataclass(frozen=True)
class ARGS_alsc_convergence:
  verbose : int = 0
  Nx : int = 100
  Ny : int = 10
  Ny_decay : bool = True
  n_param : int = 200
  c_offset : float = 1
  c_var : float = 4
  c_decay : float = 2
  tree_type : str = 'linear'
  cross_niter : int = 15
  cross_eps : float = 1e-4
  cross_kickrank : int = 5
  alsc_niter : int = 5
  alsc_kickrank : int = 5
  alsc_rinit : int = 0
  rng_seed : int = 0
  N_mc : int = 100

def RMS(x, **kwargs):
  return np.sqrt(np.mean(np.square(x), **kwargs))

def alsc_convergence(args : ARGS_alsc_convergence):
  stats = {}
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  cfun = lambda x : torch.from_numpy(cfun_np(x))
  PDE_fun = Diffusion1D(args.Nx)

  rng = np.random.default_rng(seed=args.rng_seed)

  if args.Ny_decay:
    param_shape = np.ceil(args.Ny *  np.log(2) / np.log(np.arange(2, args.n_param+2)))
    param_shape = param_shape[param_shape > 1]
    param_shape = tuple(int(ny) for ny in param_shape)
  else:
    param_shape = tuple([args.Ny] * args.n_param)
  print(param_shape)

  if args.tree_type == 'linear':
    tree = Tree.from_tupletree(linear_tupletree(args.n_param))
  elif args.tree_type == 'balanced':
    tree = Tree.from_tupletree(balanced_binary_tupletree(args.n_param))
  elif args.tree_type == 'weighted':
    w = (np.arange(args.n_param)+1)**-args.c_decay
    tree = Tree.from_tupletree(weighted_binary_tupletree(args.n_param, w))

  if args.verbose: print('Computing coefficient TT approximation')
  C_a = TreeBasedTensor.randn(tree, (args.Nx,) + param_shape, args.Ny, dtype=torch.float64, seed=args.rng_seed)
  cross = TreeCross(C_a)
  cross.run(
    cfun, 
    n_iter = args.cross_niter, 
    eps = args.cross_eps, 
    kickrank = args.cross_kickrank, 
    verbose = args.verbose
    )
  stats['cross_neval'] = cross.n_eval
  C_a.round(tol=args.cross_eps)

  # coeff TT error estimate
  errs = np.zeros(args.N_mc)
  for i in range(args.N_mc):
    y = rng.integers(0, param_shape) # random index in param space
    C_eval = torch.ravel(C_a[:, *y]) # eval in full physical space
    X = np.hstack([np.arange(args.Nx).reshape((-1,1)), np.tile(y, [args.Nx,1])])
    C_true = cfun(X)
    errs[i] = RMS((C_eval-C_true).numpy(force=True))

  stats['coeff_error'] = errs
  if args.verbose:  print(f'Coefficient TT RMS l2 err = {RMS(errs):.2e}')

  # helper tensor that is constant 1
  C_const = TreeBasedTensor.ones(tree, (1,) + param_shape, dtype=torch.float64)

  alsc = TreeALSCross(
    [C_a, C_const],
    [C_const],
    PDE_fun,
    rinit=args.alsc_rinit,
    kickrank=args.alsc_kickrank,
    verbose=args.verbose,
  )

  stats['alsc_n_eval'] = []
  stats['error'] = []
  stats['error_coeff'] = []
  stats['size'] = []

  for iter in range(args.alsc_niter):
    alsc.run(n_iter=1)
    u = alsc.get_tensor()

    stats['alsc_n_eval'] += [alsc.n_eval]
    stats['size'] += [u.sparse_size]

    errs = np.zeros(args.N_mc)
    errs_coeff = np.zeros(args.N_mc)
    for n in range(args.N_mc):
      # get random parameter
      y = rng.integers(0, param_shape)
      X = np.hstack([np.arange(args.Nx).reshape((-1,1)), np.tile(y, [args.Nx,1])])

      # reference solution
      C_a_ref = cfun(X)
      U = PDE_fun.solve([C_a_ref.reshape(-1,1), torch.zeros((1,1))], [torch.zeros((1,1))]) # solve PDE
      U = np.ravel(U)

      # eval ALS-cross
      errs[n] = RMS(U - np.ravel(u[:, *y]))

      # reference solution for coefficient approximation
      C_eval = np.ravel(C_a[:, *y[:]])
      U_c = PDE_fun.solve([torch.from_numpy(C_eval).reshape(-1,1), torch.zeros((1,1))], [torch.zeros((1,1))]) # solve PDE
      errs_coeff[n] = RMS(U - np.ravel(U_c))

    stats['error'] += [errs]
    stats['error_coeff'] += [errs_coeff]

  fn = f'data/stats_{hash(args):X}.pkl'
  with open(fn, mode='wb') as file:
    pickle.dump((args, stats), file)
  if args.verbose: print(f'Saved stats as {fn}') 
  return stats

