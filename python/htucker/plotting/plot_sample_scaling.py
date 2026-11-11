"""
Cost-accuracy plot of the sample scaling benchmark (benchmarks/ttt_sample_scaling.py).

Relative error vs the number of PDE solves, vs wall time and vs the parametric complexity (number of core
entries of the returned tensor; there also the reference solution rounded to every eps of the ladder, i.e. the
best error at its size, and the ALS cross solutions rounded to eps). Shown are tree tensor training with
(eps, ridge) chosen by validation and by the oracle, the trunc strategy (ridge chosen by validation), ALS cross,
every single training run in gray, and optionally the validation choice of a baseline run (e.g. without
keep_best and ridge). A strategy that selects by validation has to train every grid configuration, so its wall
time is the data generation (once, the samples are shared) plus the training time of all of them.

Run from python/htucker:
  python -m plotting.plot_sample_scaling                                   # latest summary, to plot_out/sample_scaling_d<d>.png
  python -m plotting.plot_sample_scaling --file data/ttt_sample_scaling/sample_scaling_<timestamp>.json \
      --baseline data/ttt_sample_scaling/sample_scaling_<timestamp>.json --out plot_out/sample_scaling_d20.png
"""
import glob
import json
import argparse
import matplotlib.pyplot as plt

# reference palette: categorical slots 1-3 (validated for all pairs), text and chrome tokens
TRAIN, ALSC, TRUNC = '#2a78d6', '#eb6834', '#1baf7a'
ADAPT = '#eda100' # slot 4, direct label (contrast relief)
BASELINE, CONTEXT = '#8c8b86', '#c9c8c2'
TEXT, TEXT_2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def load(fn):
  with open(fn) as file:
    return json.load(file)

def per_budget(d, strategy, key):
  """ Per N the run with the smallest summary[key] among the runs of strategy, and the summed training time of those runs. """
  out = []
  for N in d['args']['N_list']:
    runs = [r for r in d['runs'] if r['N'] == N and r['strategy'] == strategy and r['summary']]
    if not runs: continue
    best = min(runs, key=lambda r: r['summary'][key])
    t = best['summary']['t_data'] + sum(r['summary']['t_train'] for r in runs)
    out += [(N, best['summary']['rel_err'], t, best)]
  return out

def style_axes(ax, xlabel):
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel(xlabel, color=TEXT_2)
  ax.grid(True, which='major', color=GRID, linewidth=.8)
  ax.set_axisbelow(True)
  for side in ['top', 'right']: ax.spines[side].set_visible(False)
  for side in ['left', 'bottom']: ax.spines[side].set_color(TEXT_2)
  ax.tick_params(colors=TEXT_2, which='both')

LABEL_GAP = 11 # minimal vertical distance of end labels that would overlap [points]
LABEL_WIDTH = 90 # labels closer than this horizontally can overlap [points]

def end_label(ax, x, y, text, color, dy=0):
  """
  Direct label right of the last point at (x, y), in text ink (the colored marker beside it carries the identity).
  Created before the layout so that it gets room, moved by place_end_labels afterwards.
  """
  ann = ax.annotate(text, (x, y), xytext=(8, dy), textcoords='offset points', va='center', fontsize=9, color=TEXT,
                    bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.8))
  ax._end_labels = getattr(ax, '_end_labels', []) + [(ann, x, y, dy)]

def place_end_labels(ax):
  """ Shift the end labels of ax down where they would overlap a label placed before (by decreasing height). """
  to_pt = 72 / ax.figure.dpi
  items = []
  for ann, x, y, dy in getattr(ax, '_end_labels', []):
    px, py = ax.transData.transform((x, y)) * to_pt
    items += [(py + dy, px, py, ann)]
  placed = []
  for target, px, py, ann in sorted(items, key=lambda i: -i[0]):
    for tp, pxp in placed:
      if abs(px - pxp) < LABEL_WIDTH and tp - target < LABEL_GAP: target = min(target, tp - LABEL_GAP)
    placed += [(target, px)]
    ann.xyann = (8, target - py)

COSTS = {
  'solves': ('Error vs PDE solves', 'PDE solves'),
  'time': ('Error vs wall time', 'wall time [s] (data + all trainings of the selection)'),
  'size': ('Error vs parametric complexity', 'model parameters (core entries of the returned tensor)'),
}

def plot(d, baseline=None, out='plot_out/sample_scaling.png', previous_alsc=False):
  val = per_budget(d, 'ladder', 'val_rel_err')
  oracle = per_budget(d, 'ladder', 'rel_err')
  trunc = per_budget(d, 'trunc', 'val_rel_err')
  adapt = per_budget(d, 'adapt', 'val_rel_err') # empty for summaries without rank adaptation
  base = per_budget(baseline, 'ladder', 'val_rel_err') if baseline else []
  alsc = d['alsc']
  ladder = d['ladder']

  fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
  for ax, cost in zip(axs, COSTS):
    pick = lambda solves, time, size: {'solves': solves, 'time': time, 'size': size}[cost]
    x = lambda pts: [pick(p[0], p[2], p[3]['summary']['final_size']) for p in pts]
    y = lambda pts: [p[1] for p in pts]

    # every training run as context
    runs = [r for r in d['runs'] if r['summary']]
    ax.scatter([pick(r['N'], r['summary']['t_data'] + r['summary']['t_train'], r['summary']['final_size']) for r in runs],
               [r['summary']['rel_err'] for r in runs], s=10, color=CONTEXT, linewidths=0, zorder=1,
               label='single training run (eps, ridge)')

    if base:
      ax.plot(x(base), y(base), ':', color=BASELINE, linewidth=1.5, marker='o', markersize=4, zorder=2,
              label='training, validation choice, baseline (no keep_best, no ridge)')
    if cost == 'size':
      # best error at a given size: the reference solution rounded to eps
      # on top of everything (thin): it can coincide with the rounded ALS cross solutions
      ax.plot([l['size'] for l in ladder], [l['floor_rel_err'] for l in ladder], '--', color=TEXT, linewidth=1,
              marker='x', markersize=6, markeredgewidth=1.5, zorder=7, label='reference solution rounded to eps (best error at its size)')
    if cost != 'size': # an envelope over N, not over the size (the oracle at N=10 is the initial tensor of the largest profile)
      # open rings above the validation choice: where both choices are the same run, the ring encloses its point
      ax.plot(x(oracle), y(oracle), '--', color=TRAIN, linewidth=1, marker='o', markersize=13, markerfacecolor='none',
              markeredgecolor=TRAIN, markeredgewidth=1.3, zorder=5, label='training, oracle choice (rings, lower envelope)')
    ax.plot(x(val), y(val), '-', color=TRAIN, linewidth=2, marker='o', markersize=7, markeredgecolor='white',
            markeredgewidth=1.5, zorder=4, label='training, (eps, ridge) by validation')
    ax.plot(x(trunc), y(trunc), '-', color=TRUNC, linewidth=2, marker='s', markersize=6.5, markeredgecolor='white',
            markeredgewidth=1.5, zorder=4, label='training, trunc (ranks by validation rounding)')
    if adapt:
      ax.plot(x(adapt), y(adapt), '-', color=ADAPT, linewidth=2, marker='^', markersize=7, markeredgecolor='white',
              markeredgewidth=1.5, zorder=4, label='training, adapt (ranks grown from 1)')
    if cost == 'size' and all('round_size' in a for a in alsc):
      # the same solutions rounded to eps (same solves, the rounding time is negligible)
      ax.plot([a['round_size'] for a in alsc], [a['round_rel_err'] for a in alsc], '--', color=ALSC, linewidth=1.5,
              marker='D', markersize=6, markerfacecolor='white', markeredgecolor=ALSC, markeredgewidth=1.5, zorder=4,
              label='ALS cross, rounded to eps')
    prev = d.get('alsc_previous') if previous_alsc else None
    if cost == 'solves' and prev:
      # the curve before the settings changed (e.g. fixed sweeps before the stopping rule)
      ps = prev['settings']
      what = f'{ps["alsc_niter"]} fixed sweeps' if not ps.get('alsc_stop') else f'stop at change <= {ps["alsc_stop"]:g} eps'
      ax.plot([a['n_eval'] for a in prev['curve']], [a['rel_err'] for a in prev['curve']], ':', color=ALSC, linewidth=1.3,
              marker='D', markersize=4.5, markerfacecolor='white', markeredgecolor=ALSC, zorder=3,
              label=f'ALS cross, previous settings ({what})')
    ax_x = [pick(a['n_eval'], a['t'], a['size']) for a in alsc]
    stop = d['args'].get('alsc_stop')
    alsc_label = 'ALS cross (tolerance eps' + (f', stops at change <= {stop:g} eps)' if stop else ')')
    ax.plot(ax_x, [a['rel_err'] for a in alsc], '-', color=ALSC, linewidth=2, marker='D', markersize=6.5,
            markeredgecolor='white', markeredgewidth=1.5, zorder=4, label=alsc_label)

    style_axes(ax, COSTS[cost][1])
    ax.set_title(COSTS[cost][0], loc='left', color=TEXT, fontsize=11)

    # direct labels at the line ends; in the size panel the training lines are not monotone and the last validation
    # and trunc tensors may have the same size, so there they are identified by the legend and the N labels only
    if cost != 'size':
      end_label(ax, x(val)[-1], y(val)[-1], 'validation' if cost == 'solves' else f'validation, N={val[-1][0]}', TRAIN,
                dy=0 if cost == 'solves' else 8) # solves: stacked in the order of the points (place_end_labels)
      # only in the solves panel: in the time panel the end of trunc can be next to any validation point and its N label
      if cost == 'solves': end_label(ax, x(trunc)[-1], y(trunc)[-1], 'trunc', TRUNC)
      if cost == 'solves' and adapt: end_label(ax, x(adapt)[-1], y(adapt)[-1], 'adapt', ADAPT)
    i_last = max(range(len(alsc)), key=lambda i: ax_x[i])
    end_label(ax, ax_x[i_last], alsc[i_last]['rel_err'], 'ALS cross', ALSC)

    if cost == 'solves':
      # tolerance of every ALS cross point, left of it (the coarsest one right, it is next to the training lines)
      for i, a in enumerate(alsc):
        right = a['eps'] == max(b['eps'] for b in alsc)
        ax.annotate(f'{a["eps"]:.0e}'.replace('e-0', 'e-').replace('e+00', ''), (a['n_eval'], a['rel_err']),
                    xytext=(7, 3) if right else (-6, 0), textcoords='offset points', ha='left' if right else 'right',
                    va='center', fontsize=7.5, color=TEXT_2)
      # the smallest budget validates on two samples only
      N0, e0 = val[0][0], val[0][1]
      if e0 > 1:
        ax.annotate(f'N = {N0}: validation on {val[0][3]["summary"]["N_val"]} samples\nis noise (oracle: {oracle[0][1]:.1f})',
                    (N0, e0), xytext=(28, 45), textcoords='data', fontsize=8, color=TEXT_2, va='center',
                    arrowprops=dict(arrowstyle='-', color=TEXT_2, linewidth=.8, shrinkA=2, shrinkB=5))
    else:
      # time and size of a budget are not monotone in N, so name the budget of every point
      # (in the time panel the last one is in the end label)
      for p, px in zip(val[:-1] if cost == 'time' else val, x(val)[:-1] if cost == 'time' else x(val)):
        ax.annotate(f'N={p[0]}', (px, p[1]), xytext=(7, 0), textcoords='offset points', ha='left', va='center',
                    fontsize=7.5, color=TEXT_2, bbox=dict(boxstyle='round,pad=.1', fc='white', ec='none', alpha=.8), zorder=6)

  axs[0].set_ylabel('relative RMS error (reference set)', color=TEXT_2)
  a = d['args']
  fig.suptitle(f'Cost-accuracy tradeoff, diffusion problem (d = {a["n_param"]}, Nx = {a["Nx"]}, {a["tree_type"]} tree)',
               x=.01, ha='left', color=TEXT, fontsize=12)
  # legend over all panels (the rounded reference is only in the size panel)
  entries = {}
  for ax in axs:
    for h, l in zip(*ax.get_legend_handles_labels()): entries.setdefault(l, h)
  ncol = 4
  fig.legend(list(entries.values()), list(entries), loc='lower center', ncol=ncol, frameon=False, fontsize=9, labelcolor=TEXT)
  rows = -(-len(entries) // ncol)
  fig.tight_layout(rect=(0, .02 + .045 * rows, 1, .97)) # room for the legend rows
  fig.canvas.draw() # final axes limits and positions for the end labels
  for ax in axs: place_end_labels(ax)
  fig.savefig(out, dpi=150)
  return fig


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--file', help='summary of ttt_sample_scaling, default: the latest')
  parser.add_argument('--baseline', help='summary of a baseline run, its validation choice is shown in gray')
  parser.add_argument('--previous_alsc', action='store_true',
                      help='also show the ALS cross curve with the previous settings (alsc_previous of the summary)')
  parser.add_argument('--out', help='default: plot_out/sample_scaling_d<d>.png')
  cli = parser.parse_args()
  fn = cli.file or sorted(glob.glob('data/ttt_sample_scaling/sample_scaling_*.json'))[-1]
  d = load(fn)
  out = cli.out or f'plot_out/sample_scaling_d{d["args"]["n_param"]}.png'
  plot(d, load(cli.baseline) if cli.baseline else None, out, previous_alsc=cli.previous_alsc)
  print(f'Saved plot to: {out} (from {fn})')
