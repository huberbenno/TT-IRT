import warnings
import pytest
import torch

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
