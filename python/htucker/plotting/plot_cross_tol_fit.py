"""
Tolerance matching plot of the cross tolerance benchmark (benchmarks/cross_tol_fit.py).

Columns: the two modes, the cross approximation at eps_end rounded to eps
('round') and the cross approximation at eps rounded to eps ('cross').
Top: relative error divided by the tolerance eps (<= 1: tolerance met).
Bottom: relative error vs the number of core entries (cost-accuracy).
Every tree of the summary is a thin line, the bold line is the median over the
trees, for the current round and the previous one (round_cut). The unrounded
cross approximations are shown in gray (dotted).

The second figure (<out>_trees.png) shows the tolerance adherence of the
current round only, per tree type: linear, balanced, weighted (by decay of the
weights) and random (every seed thin, the median bold), for both modes and for
the unrounded cross approximation at eps. The third (<out>_convergence.png)
shows the sweeps and function evaluations of the cross until its test error is
below eps, and the test error per sweep of the cross at eps_end.

Without --file all summaries of data/cross_tol_fit are merged in time order, a
tree rerun later replaces its earlier result. Trees whose cross settings
(kickrank, sweeps, stopping) differ from the majority are marked in the legend,
the settings are listed below the figures.

Run from python/htucker:
  python -m plotting.plot_cross_tol_fit       # all summaries, to plot_out/cross_tol_fit(_trees).png
  python -m plotting.plot_cross_tol_fit --file data/cross_tol_fit/summary_<timestamp>.json [...] --out <png>
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_sample_scaling import style_axes, TEXT, TEXT_2, BASELINE, CONTEXT

NEW, OLD = '#2a78d6', '#eb6834' # categorical slots 1, 2 of the reference palette
MODES = [('round', 'cross at eps_end, rounded to eps'), ('cross', 'cross at eps, rounded to eps')]
# tree types: categorical slots 2-4, weighted trees an ordinal ramp of slot 1 (blue steps 250-650) by decay
LINEAR, BALANCED, RANDOM = '#eb6834', '#1baf7a', '#eda100'
WEIGHTED_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
TREE_PANELS = [('round', 'cross at eps_end, rounded to eps'), ('cross_rounded', 'cross at eps, rounded to eps'),
               ('cross', 'cross at eps, unrounded')]
VERSIONS = [('round (orthogonalized)', NEW, {'round': 'round', 'cross': 'cross_rounded'}),
            ('round, previous', OLD, {'round': 'round_old', 'cross': 'cross_rounded_old'})]


def load(fn):
  with open(fn) as file:
    return json.load(file)

def load_runs(files):
  """ Stats files of the summaries by tree name, later summaries replace earlier ones. """
  runs = {}
  for fn in files:
    runs.update({name: load(f) for name, f in load(fn)['files'].items()})
  return runs

def cross_settings(args) -> str:
  if 'cross_svd_tol' not in args:
    trunc = 'truncation .1 eps'
  elif args['cross_svd_tol'] is None:
    trunc = 'truncation 10 machine eps'
  else:
    trunc = f'truncation {args["cross_svd_tol"]:.0e}'
  if args.get('cross_n_test', 0) > 0:
    return (f'kickrank {args["cross_kickrank"]}, {trunc}, stopped at test error < eps '
            f'(at most {args["cross_niter"]} sweeps)')
  return f'kickrank {args["cross_kickrank"]}, {trunc}, {args["cross_niter"]} sweeps'

SETTINGS_MARK = '†'

def settings_note(runs):
  """ Marked tree names (settings differ from the majority) and the footnote listing the settings. """
  settings = {n: cross_settings(d['args']) for n, d in runs.items()}
  values = list(settings.values())
  majority = max(set(values), key=values.count)
  marked = {n for n, v in settings.items() if v != majority}
  note = f'Cross: {majority}'
  if marked:
    groups = {}
    for n in sorted(marked): groups.setdefault(settings[n], []).append(n)
    note += ''.join(f'; {SETTINGS_MARK} {", ".join(ns)}: {v}' for v, ns in groups.items())
  return marked, note + '.'

def series(stats, key, field):
  return np.array([entry[field] for entry in stats[key]], dtype=float)

def median_line(ax, xs, ys, color, label):
  """ Median over the trees, per eps (the x of the cost plot is the median size). """
  ax.plot(np.median(xs, axis=0), np.median(ys, axis=0), color=color, lw=2, marker='o', ms=5, label=label,
          zorder=4, markeredgecolor='white', markeredgewidth=1)

def plot(runs, out):
  stats = list(runs.values())
  eps = np.array(stats[0]['stats']['eps'])
  fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), sharey='row')

  for col, (mode, title) in enumerate(MODES):
    ax_tol, ax_cost = axs[0, col], axs[1, col]
    for label, color, keys in VERSIONS:
      key = keys[mode]
      errs = np.array([series(d['stats'], key, 'rel_err') for d in stats])
      sizes = np.array([series(d['stats'], key, 'size') for d in stats])
      for e, s in zip(errs, sizes):
        ax_tol.plot(eps, e / eps, color=color, lw=.8, alpha=.35, zorder=2)
        ax_cost.plot(s, e, color=color, lw=.8, alpha=.35, zorder=2)
      median_line(ax_tol, np.tile(eps, (len(errs), 1)), errs / eps, color, label)
      median_line(ax_cost, sizes, errs, color, label)

    # the unrounded cross approximations: at every eps ('cross'), or the one at eps_end ('round')
    cross_errs = np.array([series(d['stats'], 'cross', 'rel_err') for d in stats])
    cross_sizes = np.array([series(d['stats'], 'cross', 'size') for d in stats])
    if mode == 'cross':
      ax_tol.plot(eps, np.median(cross_errs, axis=0) / eps, color=BASELINE, ls=':', lw=1.5, marker='s', ms=4,
                  label='cross, unrounded (median)', zorder=3)
      ax_cost.plot(np.median(cross_sizes, axis=0), np.median(cross_errs, axis=0), color=BASELINE, ls=':', lw=1.5,
                   marker='s', ms=4, label='cross, unrounded (median)', zorder=3)
    else:
      floor = np.median(cross_errs[:, -1])
      ax_tol.plot(eps, floor / eps, color=BASELINE, ls=':', lw=1.5,
                  label='error of the cross at eps_end (median)', zorder=3)
      ax_cost.axhline(floor, color=BASELINE, ls=':', lw=1.5, label='error of the cross at eps_end (median)')

    ax_tol.axhline(1, color=TEXT_2, lw=1, zorder=1)
    ax_tol.annotate('error = eps', (eps.max(), 1), xytext=(0, 4), textcoords='offset points', fontsize=9,
                    color=TEXT_2, ha='right')
    ax_tol.set_title(title, color=TEXT, fontsize=11, loc='left')
    style_axes(ax_tol, 'tolerance eps')
    style_axes(ax_cost, 'number of core entries')
  axs[0, 0].set_ylabel('relative error / eps', color=TEXT_2)
  axs[1, 0].set_ylabel('relative error', color=TEXT_2)
  for ax in axs[:, 1]: ax.tick_params(labelleft=True)
  axs[0, 0].set_ylim(1e-2, 1e1) # the error of the cross at eps_end leaves the range for small eps
  for ax in axs[0]: ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc='lower left')
  for ax in axs[1]: ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc='upper right')

  d0 = stats[0]['args']
  fig.suptitle(f'Cross tolerance matching, d = {d0["n_param"]}, {len(runs)} trees, '
               f'relative MC error ({d0["N_mc"]} samples)', color=TEXT, fontsize=12)
  fig.text(.01, .01, settings_note(runs)[1], fontsize=8, color=TEXT_2, wrap=True)
  fig.tight_layout(rect=(0, .03, 1, 1))
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.savefig(out, dpi=150)
  print(f'Saved {out}')

def tree_type(name):
  return name.split('_')[0]

def tree_lines(ax, runs, xy, marked=()):
  """
  One line per tree, x, y = xy(name), colored by tree type: random trees thin
  with their median bold (if they share x), weighted trees by decay, balanced
  and linear. Returns the legend handles, sorted by the Colless index.
  """
  mark = lambda names: f' {SETTINGS_MARK}' if any(n in marked for n in names) else ''
  random = [n for n in runs if tree_type(n) == 'random']
  weighted = sorted((n for n in runs if tree_type(n) == 'weighted'), key=lambda n: runs[n]['args']['tree_weight_decay'])
  ramp = [WEIGHTED_RAMP[i] for i in np.round(np.linspace(0, len(WEIGHTED_RAMP) - 1, len(weighted))).astype(int)]
  colless = {n: d['stats']['colless_index'] for n, d in runs.items()}
  style = dict(marker='o', ms=5, markeredgecolor='white', markeredgewidth=1)
  lines = [] # (Colless index, handle)
  for n in random:
    ax.plot(*xy(n), color=RANDOM, lw=.8, alpha=.5, zorder=2)
  xs = [xy(n)[0] for n in random]
  if random and all(len(x) == len(xs[0]) and np.allclose(x, xs[0]) for x in xs):
    x, med = xs[0], np.median([xy(n)[1] for n in random], axis=0)
    cl = [colless[n] for n in random]
    h, = ax.plot(x, med, color=RANDOM, lw=2, zorder=3, **style,
                 label=f'random, {len(random)} seeds (median), C {min(cl)}-{max(cl)}{mark(random)}')
    lines += [(np.median(cl), h)]
    i = np.argmin(x)
    ax.annotate('random', (x[i], med[i]), xytext=(-8, 0), textcoords='offset points', ha='right', va='center',
                fontsize=9, color=TEXT)
  for n, color in zip(weighted, ramp):
    h, = ax.plot(*xy(n), color=color, lw=1.5, zorder=4, **{**style, 'marker': 'D', 'ms': 4.5},
                 label=f'weighted, decay {runs[n]["args"]["tree_weight_decay"]:.3g}, C {colless[n]}{mark([n])}')
    lines += [(colless[n], h)]
  for n, color in [('balanced', BALANCED), ('linear', LINEAR)]:
    if n not in runs: continue
    h, = ax.plot(*xy(n), color=color, lw=2, label=f'{n}, C {colless[n]}{mark([n])}', zorder=5,
                 **{**style, 'marker': 's'})
    lines += [(colless[n], h)]
  return [h for _, h in sorted(lines, key=lambda x: x[0])]

def unconverged(runs):
  """ (tree, eps) of the crosses with the test error stopping that hit the sweep limit above eps. """
  out = []
  for n, d in runs.items():
    if d['args'].get('cross_n_test', 0) <= 0: continue
    for eps, c in zip(d['stats']['eps'], d['stats']['cross']):
      if c['sweeps'] >= d['args']['cross_niter'] and c['test_errs'][-1] >= eps: out += [(n, eps)]
  return out

def tree_legend(ax, handles):
  ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=TEXT_2, loc='upper left',
            bbox_to_anchor=(1.02, 1), title='tree, Colless index C', title_fontsize=9, alignment='left')

def save(fig, out):
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.savefig(out, dpi=150)
  print(f'Saved {out}')

def plot_trees(runs, out):
  """ Relative error / eps per tree (TREE_PANELS), colored by tree type. """
  marked, note = settings_note(runs)
  eps = np.array(next(iter(runs.values()))['stats']['eps'])
  fig, axs = plt.subplots(1, len(TREE_PANELS), figsize=(17, 5), sharey=True)
  for ax, (key, title) in zip(axs, TREE_PANELS):
    handles = tree_lines(ax, runs, lambda n: (eps, series(runs[n]['stats'], key, 'rel_err') / eps), marked)
    ax.axhline(1, color=TEXT_2, lw=1, zorder=1)
    ax.annotate('error = eps', (eps.max(), 1), xytext=(0, 4), textcoords='offset points', fontsize=9,
                color=TEXT_2, ha='right')
    ax.set_title(title, color=TEXT, fontsize=11, loc='left')
    style_axes(ax, 'tolerance eps')
    ax.set_xlim(eps.min() / 100, eps.max() * 3) # room for the direct label left of the first point
  axs[0].set_ylabel('relative error / eps', color=TEXT_2)
  for ax in axs[1:]: ax.tick_params(labelleft=True)
  tree_legend(axs[-1], handles)

  d0 = next(iter(runs.values()))['args']
  fig.suptitle(f'Tolerance adherence by tree type, d = {d0["n_param"]}, '
               f'relative MC error ({d0["N_mc"]} samples)', color=TEXT, fontsize=12)
  stuck = unconverged(runs)
  stuck_note = ''
  if stuck:
    by_eps = {}
    for n, e in stuck: by_eps.setdefault(e, []).append(n)
    stuck_note = 'Not converged within the sweep limit: ' + '; '.join(
      f'eps {e:.1e}: {", ".join(ns)}' for e, ns in sorted(by_eps.items())) + '. '
  fig.text(.01, .045, stuck_note + 'Above 1: the cross approximation itself is less accurate than eps; '
           'the bound of round, ||u - round(u)|| <= eps ||u||, is relative to the cross.', fontsize=8.5, color=TEXT_2)
  fig.text(.01, .01, note, fontsize=8.5, color=TEXT_2)
  fig.tight_layout(rect=(0, .07, 1, 1))
  save(fig, out)

def plot_convergence(runs, out):
  """
  Cost of the cross approximation until convergence per tree: sweeps and
  function evaluations vs eps, and the test error after every sweep of the
  cross at eps_end. Only runs with the test error stopping (cross_n_test > 0).
  """
  runs = {n: d for n, d in runs.items() if d['args'].get('cross_n_test', 0) > 0}
  if not runs:
    print('No runs with the test error stopping, convergence plot skipped')
    return
  marked, note = settings_note(runs)
  eps = np.array(next(iter(runs.values()))['stats']['eps'])
  eps_end = eps.min()
  cross = lambda n: runs[n]['stats']['cross']
  fig, axs = plt.subplots(1, 3, figsize=(17, 5))

  handles = tree_lines(axs[0], runs, lambda n: (eps, series(runs[n]['stats'], 'cross', 'sweeps')), marked)
  style_axes(axs[0], 'tolerance eps')
  axs[0].set_yscale('linear')
  axs[0].set_ylim(bottom=0)
  niter = max(d['args']['cross_niter'] for d in runs.values())
  axs[0].axhline(niter, color=TEXT_2, lw=1, ls='--', zorder=1)
  axs[0].annotate('sweep limit (not converged)', (eps.max() * 3, niter), xytext=(-2, -4), textcoords='offset points',
                  fontsize=9, color=TEXT_2, ha='right', va='top')
  axs[0].set_ylabel('sweeps until the test error < eps', color=TEXT_2)
  axs[0].set_title('sweeps', color=TEXT, fontsize=11, loc='left')

  tree_lines(axs[1], runs, lambda n: (eps, series(runs[n]['stats'], 'cross', 'n_eval')), marked)
  style_axes(axs[1], 'tolerance eps')
  axs[1].set_ylabel('function evaluations', color=TEXT_2)
  axs[1].set_title('function evaluations', color=TEXT, fontsize=11, loc='left')

  def history(n):
    errs = np.array(cross(n)[-1]['test_errs'], dtype=float)
    return np.arange(1, len(errs) + 1), errs
  tree_lines(axs[2], runs, history, marked)
  style_axes(axs[2], 'sweep')
  axs[2].set_xscale('linear')
  axs[2].axhline(eps_end, color=TEXT_2, lw=1, zorder=1)
  axs[2].annotate(f'eps = {eps_end:.0e}', (axs[2].get_xlim()[0], eps_end), xytext=(4, -4), textcoords='offset points',
                  fontsize=9, color=TEXT_2, ha='left', va='top')
  axs[2].set_ylabel('relative test error', color=TEXT_2)
  axs[2].set_title(f'convergence of the cross at eps = {eps_end:.0e}', color=TEXT, fontsize=11, loc='left')
  for ax in axs[:2]: ax.set_xlim(eps.min() / 100, eps.max() * 3)
  tree_legend(axs[-1], handles)

  d0 = next(iter(runs.values()))['args']
  fig.suptitle(f'Cross approximation until convergence by tree type, d = {d0["n_param"]}', color=TEXT, fontsize=12)
  fig.text(.01, .01, note + f' Test error: relative RMS error at {d0["cross_n_test"]} new random entries per sweep.',
           fontsize=8.5, color=TEXT_2)
  fig.tight_layout(rect=(0, .04, 1, 1))
  save(fig, out)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--file', nargs='+', help='summaries of cross_tol_fit, later ones replace trees of earlier '
                                                 'ones, default: all in time order')
  parser.add_argument('--out', default='plot_out/cross_tol_fit.png')
  opts = parser.parse_args()
  files = opts.file or sorted(glob.glob('data/cross_tol_fit/summary_*.json'))
  print('Plotting ' + ', '.join(files))
  runs = load_runs(files)
  plot(runs, opts.out)
  plot_trees(runs, opts.out.removesuffix('.png') + '_trees.png')
  plot_convergence(runs, opts.out.removesuffix('.png') + '_convergence.png')
