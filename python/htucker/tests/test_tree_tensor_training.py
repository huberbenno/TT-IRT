import warnings
import pytest
import torch
from dataclasses import replace

from test_explicit_grad import TREES, setup
from tree_tensor_training.training import ARGS_tree_tensor_training, LRSchedule, perturb, train


@pytest.mark.parametrize('noise', [0., 1e-1])
def test_perturb_relative_noise(noise):
  tree, tensor, _, _ = setup(TREES['balanced'], rank=4)
  res = perturb(tensor, noise, seed=0)
  for x, y in zip(tensor.cores, res.cores):
    assert x.shape == y.shape
    rel = torch.sqrt(torch.mean((y - x)**2) / torch.mean(x**2)).item()
    # sample std over at least 4 entries, so only roughly noise
    assert rel == 0 if noise == 0 else .1 * noise < rel < 3 * noise

def lr_history(losses, **kwargs):
  schedule = LRSchedule(1., **kwargs)
  return [schedule.step(l) for l in losses]

def test_lr_schedule_keeps_lr_while_improving():
  assert lr_history([10. / (k + 1) for k in range(30)], patience=3) == 30 * [1.]

def test_lr_schedule_reduces_every_patience_epochs_on_plateau():
  # first loss sets the best, then patience epochs without improvement per reduction
  lrs = lr_history(13 * [1.], patience=4, factor=.5)
  assert lrs == 4 * [1.] + 4 * [.5] + 4 * [.25] + [.125]

def test_lr_schedule_improvement_resets_count():
  # an improvement in between restarts the count, improvements below the threshold do not
  lrs = lr_history([1., 1., 1., .5, .5, .5, .4999, .5], patience=3, threshold=1e-2)
  assert lrs == 6 * [1.] + 2 * [.5]

@pytest.mark.parametrize('optimizer, init', [('lstsq', 'uniform'), ('lstsq', 'cross'), ('lstsq', 'cross_tensor'), ('lstsq', 'mean'), ('adam', 'mean')])
def test_train_smoke(optimizer, init):
  args = ARGS_tree_tensor_training(optimizer=optimizer, init=init, Nx=5, Ny=3, Ny_decay=False, n_param=3, N_train=50, N_test=20,
                                   rinit=2, epoch_outer=3, epoch_inner=2, N_mc=5)
  stats, tensor = train(args, return_tensor=True)
  if init in ('cross', 'cross_tensor'):
    assert stats['init_size'] == stats['cross_size']
  if init in ('cross', 'cross_tensor', 'mean'):
    assert stats['cross_MC_rms'] > 0
  for key in ['train_loss', 'test_loss', 'size', 'max_rank', 'lr', 't_sweep', 't_round', 't_eval', 't_elapsed']:
    assert len(stats[key]) == stats['n_epochs'] == 3, key
  assert stats['t_train'] >= stats['t_elapsed'][-1] # plus the final rounding
  assert stats['final_train_loss'] > 0 and stats['final_test_loss'] > 0
  assert stats['MC_errors'].shape == (5,)
  assert stats['final_size'] == tensor.sparse_size

def test_round_start_delays_rounding():
  args = ARGS_tree_tensor_training(init='uniform', Nx=5, Ny=3, Ny_decay=False, n_param=3, N_train=50, N_test=20, rinit=4,
                                   epoch_outer=3, epoch_inner=2, N_mc=5, rounding=True, round_start=10, eps=1e-1)
  stats, tensor = train(args, return_tensor=True)
  assert len(set(stats['size'])) == 1 # not rounded during training
  assert stats['final_size'] == tensor.sparse_size < stats['size'][-1] # but at the end

@pytest.mark.parametrize('n_param, cross_kickrank, cross_eps, warns', [
  (5, 0, 1e-12, True), # ranks cannot grow, so the tolerance is missed by far
  (3, 5, 1e-4, False), # small enough to be approximated exactly
])
def test_warning_if_cross_approximation_misses_tolerance(n_param, cross_kickrank, cross_eps, warns):
  args = ARGS_tree_tensor_training(init='cross', Nx=5, Ny=3, Ny_decay=False, n_param=n_param, N_train=50, N_test=20,
                                   epoch_outer=1, epoch_inner=1, N_mc=5, cross_niter=1, cross_kickrank=cross_kickrank,
                                   cross_eps=cross_eps)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    stats = train(args)
  assert any('cross approximation' in str(w.message) for w in caught) == warns
  assert (stats['cross_rel_err'] > 10 * cross_eps) == warns

def test_no_rounding_by_default():
  args = ARGS_tree_tensor_training(init='uniform', Nx=5, Ny=3, Ny_decay=False, n_param=3, N_train=50, N_test=20, rinit=4,
                                   epoch_outer=3, epoch_inner=2, N_mc=5, eps=1e-1)
  stats, tensor = train(args, return_tensor=True)
  assert len(set(stats['size'])) == 1 and stats['final_size'] == stats['size'][-1] == tensor.sparse_size
  assert stats['final_train_loss'] == pytest.approx(stats['train_loss'][-1])


def small_args(**kwargs):
  base = dict(Nx=6, Ny=3, Ny_decay=False, n_param=3, N_train=60, N_test=20, rinit=2, epoch_outer=3, epoch_inner=2, N_mc=5)
  return ARGS_tree_tensor_training(**(base | kwargs))

def test_diffusion_target_shape_and_budget_split():
  stats, tensor = train(small_args(target='diffusion', init='uniform', val_fraction=.25), return_tensor=True)
  assert tensor.shape == (7, 3, 3, 3) # Nx+1 solution nodes
  assert (stats['N_fit'], stats['N_val'], stats['n_samples']) == (45, 15, 60)
  assert stats['MC_errors'].shape == (5,)
  assert stats['MC_rel_rms'] == pytest.approx(stats['MC_rms'] / stats['ref_rms'])

def test_diffusion_values_solve_the_pde():
  from tree_tensor_training.training import target_values, tensor_shape
  from pde.diffusion_1d import Diffusion1D_torch
  from pde.coeff_function import coeff
  import numpy as np
  args = small_args(target='diffusion')
  Y = np.array([[0, 1, 2], [2, 2, 0]])
  U = target_values(args)(Y)
  assert U.shape == (2,) + tensor_shape(args)[:1]
  cfun = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  for y, u in zip(Y, U):
    c = cfun(np.hstack([np.arange(args.Nx).reshape((-1, 1)), np.tile(y, [args.Nx, 1])]))
    u_ref = Diffusion1D_torch(args.Nx).solve([torch.from_numpy(c).reshape(-1, 1), torch.zeros((1, 1))], [torch.zeros((1, 1))])[0]
    assert torch.allclose(u, u_ref.ravel())

def test_cross_init_rejected_for_diffusion():
  with pytest.raises(ValueError):
    train(small_args(target='diffusion', init='cross'))

def test_init_ranks_and_ref_file(tmp_path):
  from tree_tensor_training.training import build_tree, sample_data, target_values, tensor_shape
  import numpy as np
  args = small_args(target='diffusion', init='mean')
  tree = build_tree(args)
  ranks = tuple(range(1, tree.n_nodes + 1))
  ind, vals = sample_data(target_values(args), tensor_shape(args), 7, np.random.default_rng(3), 'cpu')
  torch.save({'ind': ind, 'vals': vals}, tmp_path / 'ref.pt')
  stats = train(replace(args, init_ranks=ranks, ref_file=str(tmp_path / 'ref.pt')))
  assert 'cross_size' not in stats # no cross with given ranks
  # 'mean' adds the constant: ranks + 1 (root rank stays 1)
  assert stats['init_ranks'] == [1 if node.isroot else ranks[node.id] + 1 for node in tree.node_list]
  assert stats['MC_errors'].shape == (7,)

def test_val_round_tols():
  from tree_tensor_training.training import val_round_tols
  tols = val_round_tols(1e-2)
  assert tols[0] == 0 and tols[1] == 1e-2 and tols[-1] == pytest.approx(1.) and len(tols) == 6

def test_val_rounding_fine_tunes_after_rounding():
  stats, tensor = train(small_args(target='diffusion', init='uniform', rinit=4, rounding=True, round_mode='val', eps=1e-2),
                        return_tensor=True)
  assert stats['n_epochs'] == stats['round_epoch'] + 3 # 3 epochs each phase
  tols = [c[0] for c in stats['val_round_candidates']]
  assert stats['val_round_tol'] in tols
  # sizes after the rounding are those of the chosen candidate
  chosen = stats['val_round_candidates'][tols.index(stats['val_round_tol'])]
  assert stats['size'][stats['round_epoch']] == chosen[2] == stats['final_size'] == tensor.sparse_size
  assert len(set(stats['size'][:stats['round_epoch']])) == 1 # no rounding in the first phase

@pytest.mark.parametrize('rounding', [False, True])
def test_keep_best_returns_best_validation_tensor(rounding):
  # random init with many ranks overfits few samples, so the best epoch is usually not the last one
  kw = dict(target='diffusion', init='uniform', rinit=4, N_train=20, val_fraction=.5, epoch_outer=6, lr_patience=1, lr_end=1e-9)
  if rounding: kw |= dict(rounding=True, round_mode='val', eps=1e-2)
  stats, tensor = train(small_args(keep_best=True, **kw), return_tensor=True)
  phases = [stats['test_loss']] if not rounding else [stats['test_loss'][:stats['round_epoch']], stats['test_loss'][stats['round_epoch']:]]
  assert len(stats['best_epochs']) == len(stats['start_test_loss']) == len(phases)
  # the final tensor is the best of the last phase (including its starting tensor)
  best = min([stats['start_test_loss'][-1]] + phases[-1])
  assert stats['final_test_loss'] == pytest.approx(best, rel=1e-12)
  e = stats['best_epochs'][-1]
  assert (stats['start_test_loss'][-1] if e is None else stats['test_loss'][e]) == best

def test_keep_best_off_by_default_and_ridge_changes_result():
  s0 = train(small_args(target='diffusion', init='uniform'))
  assert s0['best_epochs'] == [] and s0['final_test_loss'] == s0['test_loss'][-1]
  s1 = train(small_args(target='diffusion', init='uniform', ridge=1e-1))
  assert s1['final_train_loss'] != s0['final_train_loss']

def test_track_ref_records_error_per_epoch():
  stats, tensor = train(small_args(target='diffusion', init='uniform', keep_best=True, track_ref=True), return_tensor=True)
  assert len(stats['ref_rel_err']) == stats['n_epochs'] and len(stats['start_ref_rel_err']) == 1
  # the returned tensor is the best validation one, its error is the recorded one
  e = stats['best_epochs'][-1]
  assert stats['MC_rel_rms'] == pytest.approx(stats['start_ref_rel_err'][0] if e is None else stats['ref_rel_err'][e], rel=1e-10)


# ---------- rank adaptation ----------

def test_admissible():
  from tree.tree import Tree
  from tree_tensor_training.training import admissible
  # root -> (leaf 0, node a -> (leaf 1, leaf 2)), node ids from the tupletree
  tree = Tree.from_tupletree((0, (1, 2)))
  shape = (4, 3, 3)
  ids = {('leaf', n.dim) if n.isleaf else ('root' if n.isroot else 'a'): n.id for n in tree.node_list}
  def ranks(r0, ra, r1, r2):
    r = [1] * tree.n_nodes
    r[ids[('leaf', 0)]], r[ids['a']], r[ids[('leaf', 1)]], r[ids[('leaf', 2)]] = r0, ra, r1, r2
    return r
  assert admissible(ranks(2, 2, 2, 2), tree, shape)
  assert not admissible(ranks(2, 3, 2, 2), tree, shape) # root: r0 = 2 < ra = 3 (ra <= r0 * 1)
  assert not admissible(ranks(2, 2, 4, 2), tree, shape) # leaf 1 above its mode size 3
  assert admissible(ranks(3, 3, 3, 1), tree, shape) # node a: 3 <= 3 * 1
  assert not admissible(ranks(2, 2, 3, 1), tree, shape) # node a: leaf 1 has 3 > 2 * 1

def test_select_nodes():
  from tree.tree import Tree, NodeIndexedList
  from tree_tensor_training.training import select_nodes
  tree = Tree.from_tupletree((0, (1, 2)))
  shape = (4, 3, 3)
  ranks = [1] * tree.n_nodes
  sigma = NodeIndexedList(tree.n_nodes * [None])
  # eta: second singular values, leaves 1 and 2 have the largest ones (admissible to raise alone)
  eta = {}
  for n in tree.node_list:
    if n.isroot: continue
    e = 0.5 if (n.isleaf and n.dim in (1, 2)) else 0.1
    sigma[n] = torch.tensor([1., e], dtype=torch.float64)
    eta[n.id] = e
  selected, new_ranks, got = select_nodes(sigma, ranks, tree, shape, theta_star=.8)
  assert got == eta
  big = sorted(i for i, e in eta.items() if e == 0.5)
  assert selected == big and all(new_ranks[i] == 2 for i in big)
  # with a single large eta (not admissible alone) theta decreases until all candidates are raised
  leaf0 = next(n.id for n in tree.node_list if n.isleaf and n.dim == 0)
  for n in tree.node_list:
    if not n.isroot: sigma[n] = torch.tensor([1., 0.5 if n.id == leaf0 else 0.1], dtype=torch.float64)
  selected, new_ranks, _ = select_nodes(sigma, ranks, tree, shape, theta_star=.8)
  assert selected == sorted(n.id for n in tree.node_list if not n.isroot)

def test_rank_adaptation_returns_best_iterate():
  stats, tensor = train(small_args(target='diffusion', rank_adapt=True, adapt_iter=4, adapt_epochs=5, keep_best=True,
                                   N_train=200, val_fraction=.25), return_tensor=True)
  adapt = stats['adapt']
  assert adapt[0]['m'] == 0 and max(adapt[0]['ranks']) == 1 # starts at rank 1
  assert all(sum(b['ranks']) > sum(a['ranks']) for a, b in zip(adapt, adapt[1:])) # ranks grow
  best = min(adapt, key=lambda e: e['val_loss'])
  assert stats['adapt_best'] == best['m']
  assert stats['final_ranks'] == best['ranks'] == [c.shape[-1] for c in tensor.cores]
  assert stats['final_test_loss'] == pytest.approx(best['val_loss'], rel=1e-12)

def test_rank_adaptation_and_rounding_rejected():
  with pytest.raises(ValueError):
    train(small_args(target='diffusion', rank_adapt=True, rounding=True))

def test_rank_adaptation_ignores_init():
  # init is ignored (no cross approximation), but must still be a known one
  stats = train(small_args(target='diffusion', rank_adapt=True, init='cross', adapt_iter=1, adapt_epochs=2))
  assert 'cross_neval' not in stats and max(stats['init_ranks']) == 1
  with pytest.raises(ValueError):
    train(small_args(rank_adapt=True, init='unknown'))

def test_enrich():
  import numpy as np
  from tree_tensor.tree_tensor_torch import TreeBasedTensor
  from tree_tensor_training.training import enrich, build_tree, tensor_shape
  args = small_args(target='diffusion', n_param=4)
  tree, shape = build_tree(args), tensor_shape(args)
  x = TreeBasedTensor.randn(tree, shape, 2, seed=0)
  node = next(n for n in tree.node_list if not n.isleaf and not n.isroot)
  selected = [node.id, node.children[0].id] # a node and one of its children
  y = enrich(x, selected, noise=1e-3, seed=0)
  for n in tree.node_list:
    if n.isroot: continue
    assert y.cores[n].shape[-1] == x.cores[n].shape[-1] + (n.id in selected)
  X, Y = x.numpy(), y.numpy()
  assert 0 < np.linalg.norm(Y - X) / np.linalg.norm(X) < 1e-4 # O(noise^2)
  assert np.allclose(x.numpy(), X) # x unchanged

def test_rank_adaptation_enrich():
  stats, tensor = train(small_args(target='diffusion', rank_adapt=True, adapt_init='enrich', adapt_iter=4,
                                   adapt_epochs=5, N_train=200, val_fraction=.25), return_tensor=True)
  adapt = stats['adapt']
  assert len(adapt) >= 2 and all(sum(b['ranks']) > sum(a['ranks']) for a, b in zip(adapt, adapt[1:]))
  best = min(adapt, key=lambda e: e['val_loss'])
  assert stats['final_ranks'] == best['ranks'] == [c.shape[-1] for c in tensor.cores]
  with pytest.raises(ValueError):
    train(small_args(target='diffusion', rank_adapt=True, adapt_init='unknown'))

def test_enrich_drifted_gauge():
  # the edge above the parent p of the padded node v carries a direction with a tiny weight in p and a huge one in
  # the grandparent (the same tensor): without orthogonalization the random slice of p multiplies the huge one
  import numpy as np
  from tree_tensor.tree_tensor_torch import TreeBasedTensor
  from tree_tensor_training.training import enrich, build_tree, tensor_shape
  args = small_args(target='diffusion', n_param=4)
  tree, shape = build_tree(args), tensor_shape(args)
  x = TreeBasedTensor.randn(tree, shape, 2, seed=0)
  v = next(n for n in tree.node_list if not n.isroot and not n.parent.isroot)
  p, g = v.parent, v.parent.parent
  drifted = TreeBasedTensor(x)
  drifted.cores[p][..., 1] *= 1e-8
  core_g = torch.movedim(drifted.cores[g], p.child_ind, 0)
  core_g[1] *= 1e8
  drifted.cores[g] = torch.movedim(core_g, 0, p.child_ind)
  X = x.numpy()
  assert np.allclose(drifted.numpy(), X, rtol=1e-6, atol=1e-6 * np.abs(X).max())
  y = enrich(drifted, [v.id], noise=1e-3, seed=0)
  assert np.linalg.norm(y.numpy() - X) / np.linalg.norm(X) < 1e-4

def test_orthogonalize_option():
  # the tensor is in orthogonal form at the start of every sweep; training still reduces the loss
  stats, tensor = train(small_args(target='diffusion', init='uniform', orthogonalize=True, epoch_outer=5),
                        return_tensor=True)
  assert stats['train_loss'][-1] < stats['train_loss'][0]

@pytest.mark.parametrize('kwargs', [dict(adapt_init='truncate'), dict(ridge=1e-2), dict(ridge=1e-2, adapt_select_ridge=None)])
def test_rank_adaptation_variants(kwargs):
  stats = train(small_args(target='diffusion', rank_adapt=True, adapt_iter=3, adapt_epochs=4, N_train=200,
                           val_fraction=.25, **kwargs))
  adapt = stats['adapt']
  assert all(sum(b['ranks']) > sum(a['ranks']) for a, b in zip(adapt, adapt[1:]))
  if kwargs.get('adapt_select_ridge', 0.) is not None:
    assert len(adapt) >= 2 # the ranks grew at least once
