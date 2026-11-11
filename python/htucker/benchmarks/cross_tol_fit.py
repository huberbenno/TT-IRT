"""
Tolerance matching of the tree cross approximation of the coefficient: how the
error relates to the specified tolerance eps, for several tree structures.

For every eps on a log grid from eps_start to eps_end two approximations are
measured:
  'cross': the cross approximation run with tolerance eps, rounded to eps,
  'round': the cross approximation with tolerance eps_end, rounded to eps.
The cross run at eps_end is the last one of the grid, so both modes share it.
round has a relative tolerance (||u - round(u)|| <= eps ||u||), so the errors
are relative as well: the Monte Carlo RMS error over N_mc random parameter
samples (RMS over the physical dimension per sample), divided by the RMS of
the coefficient at the same samples. All approximations use the same samples.
For comparison, every rounding is also done with the previous implementation
of TreeBasedTensor.round (round_cut of benchmarks/round_benchmark.py: no
orthogonalization first, every edge cut with tol / sqrt(ndim)), stored as
cross_rounded_old and round_old.

The cross runs sweep by sweep (at most cross_niter) and stops once its relative
error at cross_n_test fresh random entries (n_test of TreeCross.run) is below
eps. The max_dx criterion of TreeCross.run is not used. The truncation of the
sweeps is cross_svd_tol (default: the dtype default of TreeCross.run, 10 times
the machine epsilon per factorization). Runs before this field existed
truncated with .1 * eps.

Usage (from python/htucker):
  python -m benchmarks.cross_tol_fit            # all trees, see TREES
  python -m benchmarks.cross_tol_fit --trees linear balanced
  python -m plotting.plot_cross_tol_fit         # plot the latest summary
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np
import torch

from benchmarks.round_benchmark import round_cut as round_old
from benchmarks.util import RMS, save_stats, coeff_mc_errors
from pde.coeff_function import coeff
from tree_cross.tree_cross_torch import TreeCross
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree_tensor_training.training import build_tree, coeff_shape, ranks_of, tree_indices

DATA_DIR = 'data/cross_tol_fit'


@dataclass(frozen=True)
class ARGS_cross_tol_fit:
  verbose : int = 0
  Nx : int = 100
  Ny : int = 10
  Ny_decay : bool = True
  n_param : int = 200
  c_offset : float = 1
  c_var : float = 4
  c_decay : float = 2.
  tree_type : str = 'linear'
  tree_weight_decay : float = 2.0
  cross_niter : int = 60 # maximal number of sweeps
  cross_kickrank : int = 10
  cross_n_test : int = 1000 # random entries of the error estimate after every sweep (stopping rule)
  cross_svd_tol : float = None # truncation of the sweeps (svd_tol of TreeCross.run), None: its dtype default
  eps_start : float = 1e0
  eps_end : float = 1e-7
  eps_steps : int = 5
  rng_seed : int = 0
  N_mc : int = 100


def tensor_info(tensor: TreeBasedTensor):
  return {'size': int(tensor.sparse_size), 'ranks': ranks_of(tensor)}

def run_cross(cfun, tree, shape, eps: float, args: ARGS_cross_tol_fit):
  """
  Cross approximation with tolerance eps: sweeps until the relative error at
  cross_n_test new random entries is below eps (at most cross_niter). Returns
  the tensor and its stats (n_eval, t_cross, sweeps, test_errs per sweep).
  """
  t = time.perf_counter()
  C = TreeBasedTensor.randn(tree, shape, args.Ny, dtype=torch.float64, seed=args.rng_seed)
  cross = TreeCross(C, seed=args.rng_seed)
  test_errs = []
  for sweep in range(args.cross_niter):
    cross.run(cfun, n_iter=1, eps=eps, svd_tol=args.cross_svd_tol, kickrank=args.cross_kickrank, n_test=args.cross_n_test,
              force_init=(sweep == 0), verbose=args.verbose > 1)
    test_errs += [cross.test_err]
    if cross.test_err < eps: break
  return C, {'n_eval': cross.n_eval, 't_cross': time.perf_counter() - t, 'sweeps': len(test_errs),
             'test_errs': test_errs}

def cross_tol_fit(args: ARGS_cross_tol_fit):
  """
  Run the benchmark for one tree, returns the stats (also saved as JSON):
    eps: the tolerances,
    cross: per eps the unrounded cross (n_eval, t_cross, sweeps, test_errs,
      size, ranks, rel_err),
    cross_rounded / round: per eps the rounded tensors of the two modes
      (size, ranks, rel_err, rel_errs: the relative errors per MC sample),
    cross_rounded_old / round_old: the same with the previous round,
    coeff_rms: RMS of the coefficient at the MC samples,
  and the tree indices.
  """
  if args.verbose: print(args)
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  cfun = lambda x : torch.from_numpy(cfun_np(x))
  shape = coeff_shape(args)
  tree = build_tree(args)
  stats = tree_indices(tree)
  eps_range = np.logspace(np.log10(args.eps_start), np.log10(args.eps_end), num=args.eps_steps)
  stats['eps'] = eps_range

  # the same MC samples for every tensor (a fresh generator with the same seed per call)
  mc_errors = lambda tensor: coeff_mc_errors(tensor, cfun, np.random.default_rng(args.rng_seed), args.N_mc)
  # errors of the zero tensor: the RMS of the coefficient per sample
  coeff_per_sample = mc_errors(0 * TreeBasedTensor.ones(tree, shape))
  stats['coeff_rms'] = RMS(coeff_per_sample)
  def measure(tensor):
    rel_errs = mc_errors(tensor) / stats['coeff_rms']
    return {**tensor_info(tensor), 'rel_err': RMS(rel_errs), 'rel_errs': rel_errs}

  for key in ['cross', 'cross_rounded', 'cross_rounded_old']: stats[key] = []
  for i, eps in enumerate(eps_range):
    C, cross_stats = run_cross(cfun, tree, shape, eps, args)
    stats['cross'] += [{**cross_stats, **tensor_info(C), 'rel_err': RMS(mc_errors(C)) / stats['coeff_rms']}]
    stats['cross_rounded'] += [measure(TreeBasedTensor(C).round(eps))]
    stats['cross_rounded_old'] += [measure(round_old(TreeBasedTensor(C), eps))]
    if args.verbose:
      cr, r = stats['cross'][-1], stats['cross_rounded'][-1]
      print(f'  eps {eps:.1e}: cross {cr["t_cross"]:.0f}s, {cr["sweeps"]} sweeps, n_eval {cr["n_eval"]:.2e}, '
            f'rel err {cr["rel_err"]:.2e}, rounded {r["rel_err"]:.2e} (size {r["size"]})', flush=True)

  # C is the cross at eps_end
  stats['round'] = [measure(TreeBasedTensor(C).round(eps)) for eps in eps_range]
  stats['round_old'] = [measure(round_old(TreeBasedTensor(C), eps)) for eps in eps_range]
  if args.verbose:
    for key in ['round', 'round_old']:
      print(f'  {key}: ' + ', '.join(f'{eps:.0e}: {r["rel_err"]:.2e} ({r["size"]})'
                                     for eps, r in zip(eps_range, stats[key])))

  fn = save_stats(args, stats, DATA_DIR.removeprefix('data/'))
  if args.verbose: print(f'Saved stats to: {fn}')
  return stats


# tree configurations of plots.ipynb: name -> fields of ARGS_cross_tol_fit
TREES = {
  'linear': dict(tree_type='linear'),
  'balanced': dict(tree_type='balanced'),
  **{f'weighted_{decay:g}': dict(tree_type='weighted', tree_weight_decay=float(decay))
     for decay in np.linspace(.25, 8, num=5)},
  **{f'random_{seed}': dict(tree_type='random', rng_seed=seed) for seed in range(10)},
}

def run_tree(args: ARGS_cross_tol_fit, threads: int = 0):
  """ cross_tol_fit in a worker process (with at most threads torch threads), returns the stats file. """
  if threads > 0: torch.set_num_threads(threads)
  return cross_tol_fit(args)['file']

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--trees', nargs='+', default=list(TREES), help='names in TREES')
  parser.add_argument('--n_param', type=int, default=ARGS_cross_tol_fit.n_param)
  parser.add_argument('--eps_start', type=float, default=ARGS_cross_tol_fit.eps_start)
  parser.add_argument('--eps_end', type=float, default=ARGS_cross_tol_fit.eps_end)
  parser.add_argument('--eps_steps', type=int, default=ARGS_cross_tol_fit.eps_steps)
  parser.add_argument('--N_mc', type=int, default=ARGS_cross_tol_fit.N_mc)
  parser.add_argument('--workers', type=int, default=1, help='trees run in parallel processes')
  parser.add_argument('--threads', type=int, default=0, help='torch threads per worker (0: default)')
  opts = parser.parse_args()

  base = ARGS_cross_tol_fit(verbose=1, n_param=opts.n_param, eps_start=opts.eps_start, eps_end=opts.eps_end,
                            eps_steps=opts.eps_steps, N_mc=opts.N_mc)
  summary = {'time': time.strftime('%Y%m%d-%H%M%S'), 'files': {}}
  fn_summary = f'{DATA_DIR}/summary_{summary["time"]}.json'
  os.makedirs(DATA_DIR, exist_ok=True)
  t0 = time.perf_counter()
  with ProcessPoolExecutor(opts.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
    futures = {pool.submit(run_tree, replace(base, **TREES[name]), opts.threads): name for name in opts.trees}
    for k, future in enumerate(as_completed(futures)):
      name = futures[future]
      summary['files'][name] = future.result()
      # in the order of TREES, independent of the completion order
      summary['files'] = {n: summary['files'][n] for n in opts.trees if n in summary['files']}
      with open(fn_summary, 'w') as f: json.dump(summary, f, indent=1)
      print(f'[{k + 1}/{len(opts.trees)}] {name} done after {(time.perf_counter() - t0) / 60:.1f} min', flush=True)
  print(f'Summary: {fn_summary}')
