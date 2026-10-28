"""
Benchmark tree tensor training (tree_tensor_training.training.train).

By default with the defaults of ARGS_tree_tensor_training: explicit gradients, preconditioned per-core
least squares, starting from the mean plus a random tensor with the ranks of a tree cross approximation
(whose error serves as reference). Measures per run the training time (sweep and rounding, per epoch
and total), train and test loss after every epoch and the Monte Carlo error of the trained tensor.
Every run is done in a fresh process, so the peak memory is per run. Runs with otherwise equal
arguments (e.g. --optimizer adam vs lstsq) use the same data and initial tensor.

The stats of every run are saved to data/tree_tensor_training/stats_ttt_<hash>.json and a
summary of all runs to data/tree_tensor_training/ttt_benchmark_<timestamp>.json. Summary table and
per epoch losses are printed to stdout.

Run from python/htucker:
  python -m benchmarks.tree_tensor_training_benchmark                   # default grid
  python -m benchmarks.tree_tensor_training_benchmark --n_param 20      # default grid, n_param overridden
  python -m benchmarks.tree_tensor_training_benchmark --single --n_param 50 --init uniform --rinit 10
Every field of ARGS_tree_tensor_training can be set as --<field> <value>.
"""
import os
import json
import time
import resource
import argparse
import itertools
import multiprocessing as mp
from queue import Empty
from dataclasses import asdict, fields, replace
from benchmarks.util import NumpyEncoder, save_stats

from tree_tensor_training.training import ARGS_tree_tensor_training, train


DATA_DIR = 'tree_tensor_training' # subdirectory of data
BASE_ARGS = ARGS_tree_tensor_training(verbose=1)
# add an optimizer axis last to compare optimizers, so that runs of the same config are next to each other
DEFAULT_GRID = dict(
  tree_type = ['balanced', 'linear'],
  n_param = [10, 20, 50],
)

def peak_rss_MB():
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def run(args: ARGS_tree_tensor_training):
  """ Train, save the stats and return them. """
  stats = train(args)
  stats['peak_rss_MB'] = peak_rss_MB()

  fn = save_stats(args, stats, DATA_DIR, prefix='stats_ttt')
  if args.verbose: print(f'Saved stats to: {fn}', flush=True)
  return stats

def _worker(args, queue):
  queue.put(run(args))

def run_isolated(args: ARGS_tree_tensor_training):
  """ Run in a fresh process so that the peak memory is measured per configuration. """
  ctx = mp.get_context('spawn')
  queue = ctx.Queue()
  p = ctx.Process(target=_worker, args=(args, queue))
  p.start()
  # get before join, the child only exits once its result is consumed
  stats = None
  while stats is None and p.is_alive():
    try: stats = queue.get(timeout=1)
    except Empty: pass
  if stats is None: # a small result may be sent before the child exits
    try: stats = queue.get(timeout=1)
    except Empty: pass
  p.join()
  if p.exitcode != 0:
    return None
  return stats

def summary(stats):
  return {
    'n_epochs': stats['n_epochs'],
    't_train': stats['t_train'],
    't_sweep_mean': sum(stats['t_sweep']) / stats['n_epochs'],
    't_round_mean': sum(stats['t_round']) / stats['n_epochs'],
    'train_loss': stats['final_train_loss'],
    'test_loss': stats['final_test_loss'],
    'test_loss_min': min(stats['test_loss']),
    'MC_rms': stats['MC_rms'],
    'MC_max': stats['MC_max'],
    'final_size': stats['final_size'],
    'init_size': stats['init_size'],
    'peak_rss_MB': stats['peak_rss_MB'],
    't_cross': stats.get('t_cross'),
    'cross_size': stats.get('cross_size'),
    'cross_MC_rms': stats.get('cross_MC_rms'),
  }

def _fmt(x, spec, width):
  return f'{"-":>{width}s}' if x is None else f'{x:{width}{spec}}'

def _run_label(a: ARGS_tree_tensor_training):
  init = {'cross': 'cross', 'cross_tensor': f'C+{a.init_noise:.0e}', 'mean': f'M+{a.init_noise:.0e}'}.get(a.init, f'r={a.rinit}')
  return f'{a.optimizer:>9s} {a.tree_type:>8s} {a.n_param:4d} {init:>7s}'

def print_summary(results):
  """ results: list of (args, stats or None). MC rms of cross and trained tensor are at the same samples. """
  print(f'{"optimizer":>9s} {"tree":>8s} {"d":>4s} {"init":>7s} | {"t_cross":>7s} {"t_train":>8s} {"t/sweep":>8s} {"epochs":>6s} | '
        f'{"train":>8s} {"test":>8s} {"min test":>8s} | {"cross":>8s} {"MC rms":>8s} {"MC max":>8s} | '
        f'{"cross sz":>8s} {"init sz":>8s} {"size":>8s} {"peak":>7s}')
  for a, s in results:
    head = _run_label(a) + ' | '
    if s is None:
      print(head + 'failed')
      continue
    m = summary(s)
    print(head + f'{_fmt(m["t_cross"], ".1f", 6)}s {m["t_train"]:7.1f}s {m["t_sweep_mean"]:7.3f}s {m["n_epochs"]:6d} | '
          f'{m["train_loss"]:8.2e} {m["test_loss"]:8.2e} {m["test_loss_min"]:8.2e} | '
          f'{_fmt(m["cross_MC_rms"], ".2e", 8)} {m["MC_rms"]:8.2e} {m["MC_max"]:8.2e} | '
          f'{_fmt(m["cross_size"], "d", 8)} {m["init_size"]:8d} {m["final_size"]:8d} {m["peak_rss_MB"]:5.0f}MB')

def print_losses(results, every: int = 10):
  """ Train / test loss at every `every`-th and the last epoch, two lines per run. """
  n_max = max((s['n_epochs'] for _, s in results if s is not None), default=0)
  print(f'{"epoch":>{len(_run_label(BASE_ARGS))}s}       | ' + ' '.join(f'{e:7d}' for e in range(0, n_max, every)))
  for a, s in results:
    if s is None: continue
    epochs = list(range(0, s['n_epochs'], every))
    last = f' | last ({s["n_epochs"]-1}) ' if (s['n_epochs'] - 1) % every else ''
    for key in ['train_loss', 'test_loss']:
      line = _run_label(a) + f' {key[:-5]:>5s} | ' + ' '.join(f'{s[key][e]:7.1e}' for e in epochs)
      print(line + (last + f'{s[key][-1]:7.1e}' if last else ''))

def benchmark(configs):
  """ Run all configs, print the summary and save it. """
  results = []
  for args in configs:
    results += [(args, run_isolated(args))]

  print('\n== summary (losses after the final rounding, times in s) ==')
  print_summary(results)
  print('\n== train / test loss per epoch ==')
  print_losses(results)

  fn = f'data/{DATA_DIR}/ttt_benchmark_{time.strftime("%Y%m%d-%H%M%S")}.json'
  os.makedirs(os.path.dirname(fn), exist_ok=True)
  runs = [{'args': asdict(a), 'file': s['file'] if s else None, 'summary': summary(s) if s else None} for a, s in results]
  with open(fn, mode='w') as file:
    json.dump(runs, file, cls=NumpyEncoder)
  print(f'\nSaved summary to: {fn}')
  return results

def grid_configs(base: ARGS_tree_tensor_training, grid: dict):
  keys = list(grid)
  return [replace(base, **dict(zip(keys, values))) for values in itertools.product(*grid.values())]

def parse_bool(s: str) -> bool:
  if s.lower() in ('1', 'true', 'yes'): return True
  if s.lower() in ('0', 'false', 'no'): return False
  raise argparse.ArgumentTypeError(f'not a bool: {s}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--single', action='store_true', help='run one config instead of the grid')
  for f in fields(ARGS_tree_tensor_training):
    parser.add_argument(f'--{f.name}', type=parse_bool if f.type is bool else f.type, default=None,
                        help=f'default {getattr(BASE_ARGS, f.name)}')
  cli = vars(parser.parse_args())
  single = cli.pop('single')
  overrides = {k: v for k, v in cli.items() if v is not None}

  grid = {} if single else {k: v for k, v in DEFAULT_GRID.items() if k not in overrides}
  benchmark(grid_configs(replace(BASE_ARGS, **overrides), grid))
