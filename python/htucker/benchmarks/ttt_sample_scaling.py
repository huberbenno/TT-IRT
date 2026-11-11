"""
Sample scaling of tree tensor training on the parametric 1d diffusion problem:
accuracy vs cost, with the cost counted in PDE solves (training samples),
compared with TreeALSCross (its n_eval).

Every run of N samples uses N - N_val samples for training and
N_val = max(2, val_fraction * N) for validation (stopping, keep_best and the
selections below), so its cost is exactly N solves. Errors are relative RMS
errors on a fixed reference set of N_ref exact solves, shared by all methods.
With keep_best every run returns the tensor with the best validation loss, its
initial tensor included. Every run is repeated for every ridge parameter in
ridge_list (relative Tikhonov regularization of the core solves).

Rank strategies:
  ladder: An accurate reference solution u_ref (TreeALSCross) is rounded to
          every eps in eps_list (eps = 1: rank 1), giving rank profiles r(eps).
          Every (N, eps, ridge) is trained from the mean plus a random tensor
          with ranks r(eps). Per N the summary shows the oracle choice (best
          reference error) and the validation choice (best validation loss),
          without ridge and over all ridge parameters, and the budget rule:
          the largest profile with dof <= N_fit * r0 / budget_c, where r0 is
          the rank of the physical leaf and dof = size - sum_e r_e^2 the number
          of core entries without the gauge freedom of the edges (ridge then
          chosen by validation). The rounding error of u_ref at eps is the
          approximation floor of the profile.
  trunc:  Start with the ranks r(trunc_eps) (overparametrized for small N),
          train until the schedule stops, round to the tolerance with the best
          validation loss and train again (round_mode 'val').
  adapt:  Grow the ranks from rank 1 (rank_adapt, Grelier, Nouy, Chevreuil
          2019, Sec. 4.1) and return the iterate with the best validation
          loss. Needs no reference solution.

The column eq/dof is N_fit * r0 / dof (equations per degree of freedom), the
calibration table shows it for the oracle choice, which is the budget_c the
budget rule would have needed.

ALS cross comparison: for every eps the coefficient cross approximation rounded
to eps, ALS cross with tolerance alsc_eps_extra * eps and at most alsc_niter
sweeps, stopping after the first sweep with relative change <= alsc_stop * eps
(alsc_stop = 0: alsc_niter fixed sweeps). Its solution is also rounded to eps
(keys round_*).

The reference solution and the reference set are cached in
data/ttt_sample_scaling/ (keyed by the arguments they depend on), the stats of
every run are saved there as stats_ttt_<hash>.json and the summary as
sample_scaling_<timestamp>.json.

Run from python/htucker:
  python -m benchmarks.ttt_sample_scaling
  python -m benchmarks.ttt_sample_scaling --N_list 10,100 \\
      --eps_list 1e-2,1e-4 --n_param 10 --ridge_list 0
  python -m benchmarks.ttt_sample_scaling --strategies trunc
Every field of ARGS_ttt_sample_scaling can be set as --<field> <value>, tuples
comma separated.

Maintenance of existing summaries:
  --redo_alsc SUMMARY     recompute the ALS cross curve in place, with the
                          default ALS cross settings (ALSC_FIELDS) overridden
                          from the command line; with other settings than the
                          summary's the previous curve is kept as alsc_previous
  --rerun_capped SUMMARY  rerun the runs that stopped at the epoch limit with
                          --epoch_outer (default 500), only N in --N_list if
                          given, and save a new summary with these runs replaced
  --rerun_strategy STRATEGY SUMMARY
                          rerun all runs of a strategy with the current defaults
                          of its settings (e.g. after changing them), overrides
                          from the command line, and save a new summary
  python -m benchmarks.ttt_sample_scaling --redo_alsc SUMMARY
  python -m benchmarks.ttt_sample_scaling --rerun_capped SUMMARY \\
      --epoch_outer 500 --N_list 10000
with SUMMARY e.g. data/ttt_sample_scaling/sample_scaling_<timestamp>.json.
"""
import os
import json
import time
import hashlib
import argparse
import numpy as np
import torch
from dataclasses import dataclass, asdict, fields, replace

from benchmarks.util import NumpyEncoder, RMS
from benchmarks.tree_tensor_training_benchmark import run_isolated, parse_bool
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree_als_cross.tree_als_cross_torch import TreeALSCross
from pde.diffusion_1d import Diffusion1D_torch
from tree_tensor_training.training import (ARGS_tree_tensor_training, build_tree, coeff_shape, tensor_shape,
                                           target_values, sample_data, cross_approximation, ref_errors)
from pde.coeff_function import coeff


DATA_DIR = 'ttt_sample_scaling' # subdirectory of data

@dataclass(frozen=True)
class ARGS_ttt_sample_scaling:
  verbose : int = 1 # 1: progress, 2: also the training and ALS cross output
  # problem, as in ARGS_tree_tensor_training
  Nx : int = 100
  Ny : int = 10
  Ny_decay : bool = True
  n_param : int = 20
  c_offset : float = 1
  c_var : float = 4
  c_decay : float = 2.
  tree_type : str = 'balanced'
  tree_weight_decay : float = 0.
  rng_seed : int = 0
  # sample budgets, rank ladder, regularization and strategies
  N_list : tuple = (10, 100, 1000, 10000)
  eps_list : tuple = (1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5) # 1: rank 1
  ridge_list : tuple = (0., 1e-4, 1e-3, 1e-2)
  strategies : tuple = ('ladder', 'trunc', 'adapt')
  keep_best : bool = True
  budget_c : float = 2. # budget rule: equations (N_fit * r0) per degree of freedom
  val_fraction : float = .2
  N_ref : int = 1000 # reference set (exact solves) for all errors
  # reference solution: coefficient cross, then ALS cross
  ref_cross_eps : float = 1e-8
  ref_tol : float = 1e-8
  ref_niter : int = 8
  cross_niter : int = 15
  cross_kickrank : int = 5
  alsc_kickrank : int = 5
  alsc_rinit : int = 5
  # ALS cross comparison: coefficient rounded to eps, ALS tolerance alsc_eps_extra * eps (as in alsc_accuracy_scaling)
  alsc_niter : int = 10 # maximal number of sweeps (the number of sweeps with alsc_stop = 0)
  alsc_eps_extra : float = .1
  alsc_stop : float = 1. # stop after the first sweep with relative change <= alsc_stop * eps (TreeALSCross.run stop_tol), 0: fixed sweeps
  # training
  adapt_theta : float = .8 # adapt: see ARGS_tree_tensor_training
  adapt_iter : int = 40
  adapt_epochs : int = 20
  adapt_init : str = 'enrich' # adapt: start of every fit at new ranks, 'enrich' or 'truncate'
  adapt_select_ridge : float = 0. # adapt: ridge of the fit selecting the nodes, None: the ridge of the run
  trunc_eps : float = None # trunc: initial rank profile and smallest rounding tolerance, default min(eps_list)
  epoch_outer : int = 100
  epoch_inner : int = 2
  init_noise : float = 1e-2
  device : str = 'cpu'

# fields that the cached reference set / solution depend on, and the ALS cross comparison settings (redo_alsc)
PROBLEM_FIELDS = ['Nx', 'Ny', 'Ny_decay', 'n_param', 'c_offset', 'c_var', 'c_decay', 'tree_type', 'tree_weight_decay', 'rng_seed']
REF_FIELDS = PROBLEM_FIELDS + ['ref_cross_eps', 'ref_tol', 'ref_niter', 'cross_niter', 'cross_kickrank', 'alsc_kickrank', 'alsc_rinit']
ALSC_FIELDS = ['alsc_niter', 'alsc_kickrank', 'alsc_rinit', 'alsc_eps_extra', 'alsc_stop']
# settings of the training runs besides the problem and the strategies (rerun_strategy --runs_from compares them)
TRAINING_FIELDS = ['N_ref', 'val_fraction', 'keep_best', 'epoch_outer', 'epoch_inner', 'init_noise', 'device']
# settings of a strategy (rerun_strategy resets them to the current defaults)
STRATEGY_FIELDS = {'adapt': ['adapt_theta', 'adapt_iter', 'adapt_epochs', 'adapt_init', 'adapt_select_ridge'],
                   'trunc': ['trunc_eps'], 'ladder': []}
# version of the code computing the cached reference solution, part of its file name: bump it when that code
# changes (2: TreeCross with the dtype default truncation svd_tol and the relative max_dx)
REF_VERSION = 2


#### helpers

def problem_args(args: ARGS_ttt_sample_scaling, **kwargs) -> ARGS_tree_tensor_training:
  """
  ARGS_tree_tensor_training of the problem (target diffusion), with kwargs on
  top.
  """
  return ARGS_tree_tensor_training(target='diffusion', **{k: getattr(args, k) for k in PROBLEM_FIELDS}, **kwargs)

def cache_key(args, names) -> str:
  """ Hash of the given fields, stable across processes (unlike hash()). """
  d = {k: getattr(args, k) for k in names}
  return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

def log(args, *msg):
  if args.verbose: print(*msg, flush=True)

def rel_error(tensor, ref) -> float:
  """ Relative RMS error of tensor on the reference set. """
  return RMS(ref_errors(tensor, ref['ind'], ref['vals'])) / ref['rms']

def dof(size, ranks, tree) -> int:
  """
  Number of core entries without the gauge freedom: size - sum of r_e^2 over the
  edges (all nodes but the root).
  """
  return int(size - sum(r**2 for node, r in zip(tree.node_list, ranks) if not node.isroot))

def tensor_dof(tensor: TreeBasedTensor) -> int:
  return dof(tensor.sparse_size, [core.shape[-1] for core in tensor.cores], tensor.tree)

def tensor_info(tensor: TreeBasedTensor, ref) -> dict:
  """ Size, maximal rank and relative error on the reference set. """
  return {'size': int(tensor.sparse_size), 'max_rank': max(core.shape[-1] for core in tensor.cores),
          'rel_err': rel_error(tensor, ref)}


#### summaries

def load_summary(fn):
  with open(fn) as file:
    return json.load(file)

def args_from_summary(out, **overrides) -> ARGS_ttt_sample_scaling:
  """
  The arguments of a summary (missing fields at their defaults), with overrides
  on top.
  """
  names = {f.name for f in fields(ARGS_ttt_sample_scaling)}
  saved = {k: tuple(v) if isinstance(v, list) else v for k, v in out['args'].items() if k in names}
  return replace(ARGS_ttt_sample_scaling(), **(saved | overrides))

def save_summary(out, fn: str = None) -> str:
  """
  Save a summary to fn, by default to a new
  data/<DATA_DIR>/sample_scaling_<timestamp>.json. Returns the file name.
  """
  fn = fn or f'data/{DATA_DIR}/sample_scaling_{time.strftime("%Y%m%d-%H%M%S")}.json'
  os.makedirs(os.path.dirname(fn), exist_ok=True)
  with open(fn, mode='w') as file:
    json.dump(out, file, cls=NumpyEncoder, indent=1)
  return fn


#### reference set and solution

def reference_set(args: ARGS_ttt_sample_scaling):
  """
  N_ref random parameter indices and the exact solutions there (cached). Returns
  the dict and the file.
  """
  fn = f'data/{DATA_DIR}/refset_{cache_key(args, PROBLEM_FIELDS + ["N_ref"])}.pt'
  if not os.path.exists(fn):
    log(args, f'Computing the reference set ({args.N_ref} solves)')
    pargs = problem_args(args)
    ind, vals = sample_data(target_values(pargs), tensor_shape(pargs), args.N_ref,
                            np.random.default_rng(seed=args.rng_seed + 1000), 'cpu')
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    torch.save({'ind': ind, 'vals': vals}, fn)
  ref = torch.load(fn)
  ref['rms'] = torch.sqrt(torch.mean(ref['vals']**2)).item()
  return ref, fn

def coefficient_cross(args: ARGS_ttt_sample_scaling, eps: float):
  """
  Tree cross approximation of the coefficient, rounded to eps. Returns the
  tensor and the number of evaluations.
  """
  pargs = problem_args(args, cross_eps=eps, cross_niter=args.cross_niter, cross_kickrank=args.cross_kickrank,
                       verbose=max(args.verbose - 1, 0))
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  return cross_approximation(lambda x: torch.from_numpy(cfun_np(x)), build_tree(pargs), coeff_shape(pargs), pargs)

def als_cross(args: ARGS_ttt_sample_scaling, C: TreeBasedTensor, tol: float, n_iter: int, stop_tol: float = None):
  """
  TreeALSCross solution of the diffusion problem with the coefficient C, at most
  n_iter sweeps with stop_tol. Returns the tensor, the number of solves and the
  number of sweeps.
  """
  C_const = TreeBasedTensor.ones(C.tree, (1,) + C.shape[1:], dtype=C.dtype)
  alsc = TreeALSCross([C, C_const], [C_const], Diffusion1D_torch(args.Nx), rinit=args.alsc_rinit,
                      kickrank=args.alsc_kickrank, verbose=max(args.verbose - 1, 0), seed=args.rng_seed)
  alsc.run(n_iter=n_iter, tol=tol, stop_tol=stop_tol)
  return alsc.get_tensor(), alsc.n_eval, alsc.iter_total

def reference_solution(args: ARGS_ttt_sample_scaling) -> TreeBasedTensor:
  """ Accurate ALS cross solution (cached). """
  fn = f'data/{DATA_DIR}/ref_v{REF_VERSION}_{cache_key(args, REF_FIELDS)}.pt'
  tree = build_tree(problem_args(args))
  if not os.path.exists(fn):
    log(args, 'Computing the reference solution (coefficient cross + ALS cross)')
    t = time.perf_counter()
    C, _ = coefficient_cross(args, args.ref_cross_eps)
    u, n_eval, _ = als_cross(args, C, args.ref_tol, args.ref_niter)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    torch.save({'cores': list(u.cores), 'n_eval': n_eval, 't': time.perf_counter() - t}, fn)
  return TreeBasedTensor(torch.load(fn)['cores'], tree)


#### rank ladder and ALS cross comparison

def rank_ladder(args, u_ref, ref):
  """
  Rank profiles of u_ref rounded to every eps, with the size, dof and error
  (approximation floor) of the rounded tensor.
  """
  phys_leaf = u_ref.tree.dim2leaf(0)
  ladder = []
  for eps in args.eps_list:
    u = TreeBasedTensor(u_ref).round(eps)
    ranks = [int(core.shape[-1]) for core in u.cores]
    ladder += [{'eps': eps, 'ranks': ranks, 'r0': ranks[phys_leaf.id], 'max_rank': max(ranks),
                'size': int(u.sparse_size), 'dof': tensor_dof(u), 'floor_rel_err': rel_error(u, ref)}]
  return ladder

def alsc_curve(args, ref):
  """
  ALS cross at every eps (coefficient rounded to eps, tolerance alsc_eps_extra *
  eps): solves, sweeps, time, size, maximal rank and error, and the same after
  rounding the solution to eps (keys round_*, round_t is the time of the
  rounding only).
  """
  C, _ = coefficient_cross(args, min(args.eps_list))
  curve = []
  for eps in args.eps_list:
    t = time.perf_counter()
    u, n_eval, sweeps = als_cross(args, TreeBasedTensor(C).round(eps), args.alsc_eps_extra * eps, args.alsc_niter,
                                  stop_tol=args.alsc_stop * eps if args.alsc_stop > 0 else None)
    t_alsc = time.perf_counter() - t
    t = time.perf_counter()
    u_round = TreeBasedTensor(u).round(eps)
    t_round = time.perf_counter() - t
    a = {'eps': eps, 'n_eval': n_eval, 'sweeps': sweeps, 't': t_alsc, **tensor_info(u, ref), 'round_t': t_round,
         **{f'round_{k}': v for k, v in tensor_info(u_round, ref).items()}}
    curve += [a]
    log(args, f'  ALS cross eps {eps:.0e}: {n_eval} solves ({sweeps} sweeps), rel err {a["rel_err"]:.2e}, size {a["size"]}; '
              f'rounded: rel err {a["round_rel_err"]:.2e}, size {a["round_size"]}')
  return curve


#### training runs

def rank_ladder_ranks(args, ladder, eps):
  """
  Ranks of the ladder profile of eps (computed from the reference solution if
  eps is not on the ladder).
  """
  for l in ladder:
    if l['eps'] == eps: return l['ranks']
  u = TreeBasedTensor(reference_solution(args)).round(eps)
  return [int(core.shape[-1]) for core in u.cores]

def training_configs(args: ARGS_ttt_sample_scaling, ladder, ref_file):
  """ (N, strategy, eps, ridge, ARGS_tree_tensor_training) of all runs. """
  trunc_eps = min(args.eps_list) if args.trunc_eps is None else args.trunc_eps
  trunc_ranks = rank_ladder_ranks(args, ladder, trunc_eps)
  configs = []
  for N in args.N_list:
    for ridge in args.ridge_list:
      base = problem_args(args, N_train=N, val_fraction=args.val_fraction, ref_file=ref_file, init='mean',
                          init_noise=args.init_noise, epoch_outer=args.epoch_outer, epoch_inner=args.epoch_inner,
                          keep_best=args.keep_best, ridge=ridge, device=args.device, verbose=max(args.verbose - 1, 0))
      if 'ladder' in args.strategies:
        configs += [(N, 'ladder', l['eps'], ridge, replace(base, init_ranks=tuple(l['ranks']))) for l in ladder]
      if 'trunc' in args.strategies:
        configs += [(N, 'trunc', trunc_eps, ridge, replace(base, init_ranks=tuple(trunc_ranks), rounding=True,
                                                            round_mode='val', eps=trunc_eps))]
      if 'adapt' in args.strategies:
        configs += [(N, 'adapt', 0., ridge, replace(base, rank_adapt=True, adapt_theta=args.adapt_theta,
                                                     adapt_iter=args.adapt_iter, adapt_epochs=args.adapt_epochs,
                                                     adapt_init=args.adapt_init,
                                                     adapt_select_ridge=args.adapt_select_ridge))]
  return configs

def run_summary(stats, ref_rms, tree):
  """ The summary of a training run (None if it failed). """
  if stats is None: return None
  phys_leaf = tree.dim2leaf(0)
  return {
    'rel_err': stats['MC_rel_rms'],
    'val_rel_err': np.sqrt(stats['final_test_loss']) / ref_rms,
    'train_rel_err': np.sqrt(stats['final_train_loss']) / ref_rms,
    'n_samples': stats['n_samples'], 'N_fit': stats['N_fit'], 'N_val': stats['N_val'],
    'n_epochs': stats['n_epochs'], 't_train': stats['t_train'], 't_data': stats['t_data'],
    'init_size': int(stats['init_size']), 'final_size': int(stats['final_size']),
    'final_dof': dof(stats['final_size'], stats['final_ranks'], tree),
    'final_r0': stats['final_ranks'][phys_leaf.id], 'final_max_rank': max(stats['final_ranks']),
    'best_epochs': stats['best_epochs'], 'val_round_tol': stats.get('val_round_tol'),
    'round_epoch': stats.get('round_epoch'), 'file': stats['file'],
    'adapt_iters': len(stats['adapt']) - 1 if 'adapt' in stats else None, 'adapt_best': stats.get('adapt_best'),
  }


#### selection and output

def eq_per_dof(run, ladder):
  """
  N_fit * r0 / dof of the run: of its ladder profile, or of the final tensor
  (trunc, adapt).
  """
  s = run['summary']
  if run['strategy'] == 'ladder':
    l = next(l for l in ladder if l['eps'] == run['eps'])
    return s['N_fit'] * l['r0'] / l['dof']
  return s['N_fit'] * s['final_r0'] / s['final_dof']

def select(runs, ladder, alsc, N, budget_c):
  """
  Per N: oracle and validation choice on the ladder (without ridge and over all
  ridge parameters), the budget rule (ridge by validation), trunc without ridge
  and with ridge by validation, and the best ALS cross with <= N solves.
  """
  best_err = lambda rs: min(rs, key=lambda r: r['summary']['rel_err']) if rs else None
  best_val = lambda rs: min(rs, key=lambda r: r['summary']['val_rel_err']) if rs else None
  sel = {'N': N}
  lad = [r for r in runs if r['N'] == N and r['strategy'] == 'ladder' and r['summary']]
  lad0 = [r for r in lad if r['ridge'] == 0]
  sel['oracle0'], sel['val0'] = best_err(lad0), best_val(lad0)
  sel['oracle'], sel['val'] = best_err(lad), best_val(lad)
  if lad:
    N_fit = lad[0]['summary']['N_fit']
    fits = [l for l in ladder if l['dof'] <= N_fit * l['r0'] / budget_c]
    eps = max(fits, key=lambda l: l['dof'])['eps'] if fits else min(ladder, key=lambda l: l['dof'])['eps']
    sel['budget'] = best_val([r for r in lad if r['eps'] == eps])
    # equations per dof of the oracle choice: the budget_c that would have selected it
    if sel['oracle0']: sel['oracle0_eq_per_dof'] = eq_per_dof(sel['oracle0'], ladder)
    if sel['oracle']: sel['oracle_eq_per_dof'] = eq_per_dof(sel['oracle'], ladder)
  trunc = [r for r in runs if r['N'] == N and r['strategy'] == 'trunc' and r['summary']]
  sel['trunc0'] = next((r for r in trunc if r['ridge'] == 0), None)
  sel['trunc'] = best_val(trunc)
  adapt = [r for r in runs if r['N'] == N and r['strategy'] == 'adapt' and r['summary']]
  sel['adapt0'] = next((r for r in adapt if r['ridge'] == 0), None)
  sel['adapt'] = best_val(adapt)
  affordable = [a for a in alsc if a['n_eval'] <= N]
  sel['alsc'] = min(affordable, key=lambda a: a['rel_err']) if affordable else None
  return sel

def _sel_json(v):
  """
  A selected run reduced to eps, ridge and error, other entries unchanged.
  """
  if isinstance(v, dict) and 'strategy' in v:
    return {'eps': v['eps'], 'ridge': v['ridge'], 'rel_err': v['summary']['rel_err']}
  return v

def selections(runs, ladder, alsc, args):
  """
  select for every N of args, and the same in the form saved in the summary.
  """
  sels = [select(runs, ladder, alsc, N, args.budget_c) for N in args.N_list]
  return sels, [{k: _sel_json(v) for k, v in s.items()} for s in sels]

def _e(x): return f'{"-":>8s}' if x is None else f'{x:8.2e}'

def print_tables(ladder, alsc, runs, sels):
  print('\n== rank ladder (reference solution rounded to eps) ==')
  print(f'{"eps":>7s} {"r0":>4s} {"max r":>5s} {"size":>8s} {"dof":>8s} {"floor":>8s}')
  for l in ladder:
    print(f'{l["eps"]:7.0e} {l["r0"]:4d} {l["max_rank"]:5d} {l["size"]:8d} {l["dof"]:8d} {l["floor_rel_err"]:8.2e}')

  if alsc:
    print('\n== ALS cross ==')
    print(f'{"":28s} {"before rounding":>30s} | {"rounded to eps":>30s}')
    print(f'{"eps":>7s} {"solves":>7s} {"sweeps":>6s} {"t":>5s} | {"max r":>5s} {"size":>8s} {"rel err":>8s} {"":>6s}| '
          f'{"max r":>5s} {"size":>8s} {"rel err":>8s} {"t":>6s}')
    for a in alsc:
      # older summaries have no sweeps and no rounded solution
      line = (f'{a["eps"]:7.0e} {a["n_eval"]:7d} {a.get("sweeps", "-"):>6} {a["t"]:4.1f}s | '
              f'{a["max_rank"]:5d} {a["size"]:8d} {a["rel_err"]:8.2e} {"":>6s}| ')
      if 'round_size' in a:
        line += f'{a["round_max_rank"]:5d} {a["round_size"]:8d} {a["round_rel_err"]:8.2e} {a["round_t"]:5.2f}s'
      print(line)

  print('\n== training runs (relative errors, rel err on the reference set, best: epoch of the returned tensor, - initial; '
        'adapt: best iteration / iterations) ==')
  print(f'{"N":>6s} {"strategy":>8s} {"eps":>7s} {"ridge":>6s} | {"train":>8s} {"val":>8s} {"rel err":>8s} | '
        f'{"epochs":>6s} {"best":>7s} {"t_train":>8s} | {"size":>8s} {"max r":>5s} {"eq/dof":>7s} {"tol":>7s}')
  for r in runs:
    head = f'{r["N"]:6d} {r["strategy"]:>8s} {r["eps"]:7.0e} {r["ridge"]:6.0e} | '
    s = r['summary']
    if s is None:
      print(head + 'failed')
      continue
    tol = '-' if s['val_round_tol'] is None else f'{s["val_round_tol"]:.0e}'
    if r['strategy'] == 'adapt': # best iteration / iterations
      best = f'it {s["adapt_best"]}/{s["adapt_iters"]}'
    else:
      best = ','.join('-' if e is None else str(e) for e in s['best_epochs'])
    print(head + f'{s["train_rel_err"]:8.2e} {s["val_rel_err"]:8.2e} {s["rel_err"]:8.2e} | '
          f'{s["n_epochs"]:6d} {best:>7s} {s["t_train"]:7.1f}s | {s["final_size"]:8d} {s["final_max_rank"]:5d} '
          f'{eq_per_dof(r, ladder):7.2f} {tol:>7s}')

  print('\n== per budget: relative error on the reference set (ladder: eps/ridge of the choice) ==')
  print(f'{"N":>6s} | {"oracle, ridge 0":>17s} {"val, ridge 0":>17s} | {"oracle":>17s} {"val":>17s} {"budget":>17s} | '
        f'{"trunc r 0":>9s} {"trunc val":>9s} | {"adapt r 0":>9s} {"adapt val":>9s} | {"ALS cross <= N":>22s}')
  def pick(sel, key):
    r = sel.get(key)
    return f'{"-":>17s}' if r is None else f'{r["summary"]["rel_err"]:8.2e} ({r["eps"]:.0e}/{r["ridge"]:.0e})'.replace('0e+00', '0')
  for sel in sels:
    a = sel['alsc']
    alsc_str = f'{"-":>22s}' if a is None else f'{a["rel_err"]:8.2e} ({a["n_eval"]:5d} solves)'
    trunc = lambda k: _e(sel[k]['summary']['rel_err'] if sel.get(k) else None)
    print(f'{sel["N"]:6d} | {pick(sel, "oracle0"):>17s} {pick(sel, "val0"):>17s} | {pick(sel, "oracle"):>17s} '
          f'{pick(sel, "val"):>17s} {pick(sel, "budget"):>17s} | {trunc("trunc0"):>9s} {trunc("trunc"):>9s} | '
          f'{trunc("adapt0"):>9s} {trunc("adapt"):>9s} | {alsc_str}')

  print('\n== budget rule calibration: equations per dof (N_fit r0 / dof) of the oracle choice ==')
  print(f'{"N":>6s} {"ridge 0":>8s} {"any ridge":>9s}')
  for sel in sels:
    f = lambda k: f'{sel[k]:8.2f}' if k in sel else f'{"-":>8s}'
    print(f'{sel["N"]:6d} {f("oracle0_eq_per_dof")} {f("oracle_eq_per_dof"):>9s}')


#### benchmark

def sample_scaling(args: ARGS_ttt_sample_scaling):
  log(args, args)
  ref, ref_file = reference_set(args)
  u_ref = reference_solution(args)
  ref_err = rel_error(u_ref, ref)
  log(args, f'Reference solution: rel err {ref_err:.2e}, size {u_ref.sparse_size}, '
            f'max rank {max(c.shape[-1] for c in u_ref.cores)}')
  if ref_err > min(args.eps_list):
    print(f'Warning: the reference solution (rel err {ref_err:.1e}) is less accurate than the smallest eps '
          f'{min(args.eps_list):.0e}, the finest rank profiles are not meaningful.')

  ladder = rank_ladder(args, u_ref, ref)
  log(args, 'ALS cross comparison')
  alsc = alsc_curve(args, ref)

  runs = []
  configs = training_configs(args, ladder, ref_file)
  for i, (N, strategy, eps, ridge, targs) in enumerate(configs):
    log(args, f'[{i+1}/{len(configs)}] N = {N}, {strategy}, eps {eps:.0e}, ridge {ridge:.0e}')
    stats = run_isolated(targs, data_dir=DATA_DIR)
    runs += [{'N': N, 'strategy': strategy, 'eps': eps, 'ridge': ridge, 'summary': run_summary(stats, ref['rms'], u_ref.tree)}]
    if runs[-1]['summary']: log(args, f'    rel err {runs[-1]["summary"]["rel_err"]:.2e}, {stats["t_train"]:.1f}s')

  sels, sels_json = selections(runs, ladder, alsc, args)
  print_tables(ladder, alsc, runs, sels)
  out = {'args': asdict(args), 'ref_rel_err': ref_err, 'ref_rms': ref['rms'], 'ref_file': ref_file,
         'ladder': ladder, 'alsc': alsc, 'runs': runs, 'selections': sels_json}
  print(f'\nSaved summary to: {save_summary(out)}')
  return out


#### maintenance of existing summaries

def redo_alsc(fn: str, alsc_settings: dict = None, verbose: int = 1):
  """
  Recompute the ALS cross curve of an existing summary and save it in place,
  with the default ALS cross settings (ALSC_FIELDS) overridden by alsc_settings.
  With the settings of the summary the old curve must be reproduced (ALS cross
  is seeded), otherwise the file is not changed. With other settings the old
  curve is kept as alsc_previous (with its settings) and the args are updated.
  """
  out = load_summary(fn)
  # summaries without alsc_stop used fixed sweeps
  old_settings = {k: out['args'].get(k, 0. if k == 'alsc_stop' else None) for k in ALSC_FIELDS}
  new_settings = {k: getattr(ARGS_ttt_sample_scaling(), k) for k in ALSC_FIELDS} | (alsc_settings or {})
  args = args_from_summary(out, **new_settings, verbose=verbose)
  ref, _ = reference_set(args)
  log(args, f'Recomputing the ALS cross curve of {fn} with {new_settings}')
  alsc = alsc_curve(args, ref)
  if new_settings == old_settings:
    for old, new in zip(out['alsc'], alsc):
      if old['n_eval'] != new['n_eval'] or old['size'] != new['size'] or abs(old['rel_err'] / new['rel_err'] - 1) > 1e-8:
        raise RuntimeError(f'ALS cross at eps {old["eps"]:.0e} differs from the summary, not saved: {old} vs {new}')
  else:
    out['alsc_previous'] = {'settings': old_settings, 'curve': out['alsc']}
    out['args'] |= new_settings
  out['alsc'] = alsc
  save_summary(out, fn)
  print_tables(out['ladder'], alsc, [], [])
  print(f'\nSaved the new ALS cross curve to: {fn}')

def is_capped(run, default_cap):
  """
  Whether a training phase of the run stopped at the epoch limit (instead of by
  the lr schedule).
  """
  # the limit of the run if it was rerun (rerun_capped), else that of the summary
  s, cap = run['summary'], run.get('epoch_outer', default_cap)
  if s is None or run['strategy'] == 'adapt': return False # adapt: every fit has its own cap
  if s.get('round_epoch') is None: return s['n_epochs'] >= cap
  return s['round_epoch'] >= cap or s['n_epochs'] - s['round_epoch'] >= cap

def run_key(run):
  return run['N'], run['strategy'], run['eps'], run['ridge']

def _rerun_runs(out, args, todo, extra=None, note=None):
  """
  Rerun the runs out['runs'][i], i in todo, with the training settings of args
  (in place). The record of a rerun run has its previous summary as before and
  the entries of extra. note(run) appends a remark to its log line.
  """
  ref, ref_file = reference_set(args)
  tree = build_tree(problem_args(args))
  configs = {(N, s, eps, ridge): targs for N, s, eps, ridge, targs in training_configs(args, out['ladder'], ref_file)}
  for k, i in enumerate(todo):
    r = out['runs'][i]
    log(args, f'[{k+1}/{len(todo)}] N = {r["N"]}, {r["strategy"]}, eps {r["eps"]:.0e}, ridge {r["ridge"]:.0e}: '
              f'before rel err {r["summary"]["rel_err"]:.2e}')
    stats = run_isolated(configs[run_key(r)], data_dir=DATA_DIR)
    out['runs'][i] = r | (extra or {}) | {'summary': run_summary(stats, ref['rms'], tree), 'before': r['summary']}
    s = out['runs'][i]['summary']
    if s: log(args, f'    rel err {s["rel_err"]:.2e} ({s["n_epochs"]} epochs), {s["t_train"]:.1f}s'
                    + (note(out['runs'][i]) if note else ''))

def _save_rerun(out, args, key: str, info: dict):
  """ Recompute the selections, record info under key, print the tables and save a new summary. """
  sels, out['selections'] = selections(out['runs'], out['ladder'], out['alsc'], args)
  out[key] = info
  print_tables(out['ladder'], out['alsc'], out['runs'], sels)
  print(f'\nSaved summary to: {save_summary(out)}')
  return out

def rerun_capped(fn: str, epoch_outer: int, N_list=None, verbose: int = 1):
  """
  Rerun the training runs of a summary that stopped at the epoch limit
  (optionally only those with N in N_list) with the limit epoch_outer, and save
  a new summary with these runs replaced (their records also have epoch_outer).
  All other runs stopped before the limit, so they would not change. The old
  summary is not changed.
  """
  out = load_summary(fn)
  args = args_from_summary(out, verbose=verbose)
  capped = [i for i, r in enumerate(out['runs']) if is_capped(r, args.epoch_outer) and (N_list is None or r['N'] in N_list)]
  log(args, f'Rerunning {len(capped)} runs of {fn} that stopped at the epoch limit, with epoch_outer {epoch_outer}')
  _rerun_runs(out, replace(args, epoch_outer=epoch_outer), capped, extra={'epoch_outer': epoch_outer},
              note=lambda r: '  <- still at the limit' if is_capped(r, epoch_outer) else '')
  return _save_rerun(out, args, 'rerun_capped', {'from': fn, 'epoch_outer': epoch_outer, 'runs': capped})

def rerun_strategy(fn: str, strategy: str, overrides: dict = None, runs_from: str = None, verbose: int = 1):
  """
  Rerun all runs of a strategy of a summary with the current defaults of its
  settings (STRATEGY_FIELDS) and overrides on top, and save a new summary with
  these runs replaced. With runs_from the runs are taken from that summary
  instead, which must have the same problem, training and strategy settings.
  The old summary is not changed.
  """
  out = load_summary(fn)
  settings = {k: getattr(ARGS_ttt_sample_scaling(), k) for k in STRATEGY_FIELDS[strategy]} | (overrides or {})
  args = args_from_summary(out, verbose=verbose, **settings)
  out['args'] |= {k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()}
  todo = [i for i, r in enumerate(out['runs']) if r['strategy'] == strategy]
  if runs_from is None:
    log(args, f'Rerunning the {len(todo)} {strategy} runs of {fn} with {settings}')
    _rerun_runs(out, args, todo)
  else:
    src = load_summary(runs_from)
    src_args = args_from_summary(src)
    differ = [k for k in PROBLEM_FIELDS + TRAINING_FIELDS + list(settings) if getattr(src_args, k) != getattr(args, k)]
    if differ: raise ValueError(f'{runs_from} differs from the requested settings in {differ}')
    source = {run_key(r): r['summary'] for r in src['runs'] if r['strategy'] == strategy}
    log(args, f'Taking the {len(todo)} {strategy} runs of {fn} with {settings} from {runs_from}')
    for i in todo:
      r = out['runs'][i]
      out['runs'][i] = r | {'summary': source[run_key(r)], 'before': r['summary']}
  return _save_rerun(out, args, 'rerun_strategy',
                     {'from': fn, 'strategy': strategy, 'settings': settings, 'runs_from': runs_from})


#### command line

def field_parser(f, default):
  """
  argparse type: bools as parse_bool, tuples as comma separated values of the
  type of the default entries.
  """
  if f.type is bool: return parse_bool
  if f.type is tuple:
    elem = type(default[0])
    return lambda s: tuple(elem(x) for x in s.split(','))
  if default is None: return float # trunc_eps
  return f.type

if __name__ == '__main__':
  BASE = ARGS_ttt_sample_scaling()
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--redo_alsc', metavar='SUMMARY', help='only recompute the ALS cross curve of this summary (in place)')
  parser.add_argument('--rerun_capped', metavar='SUMMARY',
                      help='rerun the runs of this summary that stopped at the epoch limit with --epoch_outer (only N in --N_list if given)')
  parser.add_argument('--rerun_strategy', nargs=2, metavar=('STRATEGY', 'SUMMARY'),
                      help='rerun all runs of a strategy of this summary with the current defaults of its settings')
  parser.add_argument('--runs_from', metavar='SUMMARY',
                      help='with --rerun_strategy: take the runs from this summary (same settings) instead of rerunning')
  for f in fields(ARGS_ttt_sample_scaling):
    default = getattr(BASE, f.name)
    parser.add_argument(f'--{f.name}', type=field_parser(f, default), default=None, help=f'default {default}')
  overrides = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
  if 'rerun_strategy' in overrides:
    strategy, fn = overrides['rerun_strategy']
    rerun_strategy(fn, strategy, {k: v for k, v in overrides.items() if k in STRATEGY_FIELDS[strategy]},
                   runs_from=overrides.get('runs_from'))
  elif 'rerun_capped' in overrides:
    rerun_capped(overrides['rerun_capped'], overrides.get('epoch_outer', 500), overrides.get('N_list'))
  elif 'redo_alsc' in overrides:
    redo_alsc(overrides['redo_alsc'], {k: v for k, v in overrides.items() if k in ALSC_FIELDS})
  else:
    sample_scaling(replace(BASE, **overrides))
