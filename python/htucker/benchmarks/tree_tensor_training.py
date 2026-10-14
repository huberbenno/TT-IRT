import numpy as np
from dataclasses import dataclass
import json
from dataclasses import asdict
from benchmarks.util import NumpyEncoder

from pde.coeff_function import coeff

import torch
from tree.tree import Tree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from tree.tree_util import balanced_binary_tupletree, linear_tupletree, weighted_binary_tupletree, random_tupletree

from torch import nn, optim
from tree.tree import Tree
from tree_tensor.tree_tensor_torch import TreeBasedTensor
from itertools import chain

class HT_Module(nn.Module):
  def __init__(self, tensor: TreeBasedTensor) -> None:
    super().__init__()
    self.tensor = TreeBasedTensor(tensor)
    for t in self.tensor.cores:
      t.requires_grad_()

  def parameters(self, recurse = True):
    if recurse:
      return chain(iter(self.tensor.cores),super().parameters(recurse))
    else:
      iter(self.tensor.cores)

  def named_parameters(self, prefix = "", recurse = True, remove_duplicate = True):
    return iter([(prefix + f'core.{i}', core) for i, core in enumerate(self.tensor.cores)])

  def forward(self, indices):
    indices = torch.atleast_2d(indices)
    val= self._forward_subtree(self.tensor.tree.root, indices)
    return val.squeeze(-1)

  def _forward_subtree(self, node, indices):
    if node.dim == 0: 
      return torch.tile(self.tensor.cores[node], (indices.shape[0], 1,1))
    if node.isleaf:
      val = self.tensor.cores[node][indices[:, node.dim]]
      return val
    else:
      n_c = node.n_children + 1
      extra_ind = []
      einsum_args = [self.tensor.cores[node], [i+1 for i in range(n_c)]]
      for i, child in enumerate(node.children):
        val_c = self._forward_subtree(child, indices)
        if len(val_c.shape) == 2:
          einsum_args += [val_c, [0, i+1]]
        elif len(val_c.shape) == 3:
          einsum_args += [val_c, [0, n_c+i+1, i+1]]
          extra_ind += [n_c+i+1]
      einsum_args += [[0] + extra_ind + [n_c]]
      eval = torch.einsum(*einsum_args)

      return eval

@dataclass(frozen=True)
class ARGS_tree_cross_training:
  verbose : int = 0
  Nx : int = 100
  Ny : int = 10
  Ny_decay : bool = True
  n_param : int = 200
  c_offset : float = 1
  c_var : float = 4
  c_decay : float = 2.
  tree_type : str = 'balanced'
  tree_weight_decay : float = 0.
  N_train : int = int(1e4)
  N_test : int = int(1e3)
  lr_start : float = 1e-2
  lr_end : float = 1e-6
  rinit : int = 20
  epoch_outer : int = 100
  epoch_inner : int = 10
  eps : float = 1e-4
  rng_seed : int = 0
  N_mc : int = 1000
  device : str = 'cpu'

def RMS(x, **kwargs):
  return np.sqrt(np.mean(np.square(x), **kwargs))

def tree_tensor_training(args: ARGS_tree_cross_training):
  print('Running tree tensor training...')
  if args.verbose: print(args)

  stats = {'train_loss': [], 'test_loss': [], 'size': [], 'lr': []}
  cfun_np = coeff(args.Nx, args.Ny, args.c_offset, args.c_var, args.c_decay)
  cfun = lambda x : torch.from_numpy(cfun_np(x))

  rng = np.random.default_rng(seed=args.rng_seed)

  if args.Ny_decay:
    tensor_shape = np.ceil(args.Ny *  np.log(2) / np.log(np.arange(2, args.n_param+2)))
    tensor_shape = tensor_shape[tensor_shape > 1]
    tensor_shape = tuple(int(ny) for ny in tensor_shape)
  else:
    tensor_shape = tuple([args.Ny] * args.n_param)

  tensor_shape = (args.Nx,) + tensor_shape

  if args.tree_type == 'linear':
    tree = Tree.from_tupletree(linear_tupletree(args.n_param))
  elif args.tree_type == 'balanced':
    tree = Tree.from_tupletree(balanced_binary_tupletree(args.n_param))
  elif args.tree_type == 'weighted':
    w = (np.arange(args.n_param)+1)**-args.tree_weight_decay
    tree = Tree.from_tupletree(weighted_binary_tupletree(args.n_param, w))
  elif args.tree_type == 'random':
    tree = Tree.from_tupletree(random_tupletree(args.n_param, seed=args.rng_seed))

  stats['sacking_index'] = np.sum(np.array(tree.get_leaf_depths()[1:])-1)
  colles_ind = 0
  subtree_dims = tree.get_subtree_dims()
  for node in tree.node_list:
    if node.isroot: continue
    if not node.isleaf and node.n_children == 2:
      colles_ind += abs(len(subtree_dims[node.children[0]]) - len(subtree_dims[node.children[1]]))

  stats['colles_index'] = colles_ind

  train_ind = []
  train_eval = []
  test_ind = []
  test_eval = []
  for i in range(args.N_train):
    y = rng.integers(0, tensor_shape)
    train_ind += [torch.from_numpy(y)]
    train_eval += [cfun(np.hstack([np.arange(args.Nx).reshape((-1,1)), np.tile(y[1:], [args.Nx,1])]))]
  train_ind = torch.vstack(train_ind).to(args.device)
  train_eval = torch.vstack(train_eval).to(args.device)

  for i in range(args.N_test):
    y = rng.integers(0, tensor_shape)
    test_ind += [torch.from_numpy(y)]
    test_eval += [cfun(np.hstack([np.arange(args.Nx).reshape((-1,1)), np.tile(y[1:], [args.Nx,1])]))]
  test_ind = torch.vstack(test_ind).to(args.device)
  test_eval = torch.vstack(test_eval).to(args.device)

  tensor = TreeBasedTensor.randn(tree, tensor_shape, args.rinit, device=args.device)
  model = HT_Module(tensor)

  lr = args.lr_start

  for epoch in range(args.epoch_outer):
    model.tensor.requires_grad_()
    loss_total = 0
    for name, param in model.named_parameters():
      print('\r' + f'{name}', end='')
      opt = optim.Adam([param], lr=lr)
      loss_local = []
      for local_epoch in range(args.epoch_inner):
        model.train()
        opt.zero_grad()
        loss = torch.tensor(0, dtype=torch.double)
        eval = model.forward(train_ind)
        loss = torch.nn.functional.mse_loss(eval, train_eval)
        loss.backward()
        loss_local += [loss.item()]
        opt.step()
        del eval, loss
      del opt
    stats['train_loss'] += [np.mean(loss_local)]  
    del loss_total

    model.tensor.round(args.eps)
    stats['size'] += [model.tensor.sparse_size]
    stats['lr'] += [lr]

    model.eval()
    with torch.no_grad():
      loss_total = 0
      eval = model.forward(test_ind)  
      loss = torch.nn.functional.mse_loss(eval, test_eval)
      loss_total += loss.item()
      stats['test_loss'] += [loss_total]

    if args.verbose:
      print('\r' + f'epoch {epoch}, train loss {stats['train_loss'][-1]:.2e}, validation loss {stats['test_loss'][-1]:.2e}, lr {lr:.1e}, size {model.tensor.sparse_size}')

    if epoch > 5 and sum(stats['test_loss'][-4:]) / 4 > 1.0 * min(stats['test_loss'][:-4]):
      lr *= .5
    if lr < args.lr_end:
      break

  # coeff TT error estimate
  errs = np.zeros(args.N_mc)
  for i in range(args.N_mc):
    y = rng.integers(0, tensor_shape[1:]) # random index in param space
    C_eval = torch.ravel(model.tensor[:, *y]) # eval in full physical space
    X = np.hstack([np.arange(args.Nx).reshape((-1,1)), np.tile(y, [args.Nx,1])])
    C_true = cfun(X)
    errs[i] = RMS((C_eval-C_true).numpy(force=True))

  stats['MC_errors'] = errs
  if args.verbose:  print(f'Coefficient TT RMS l2 err = {RMS(errs):.2e}')

  fn = f'data/stats_{hash(args):X}.json'
  stats['file'] = fn
  with open(fn, mode='w') as file:
    json.dump({'args': asdict(args), 'stats':stats}, file, cls=NumpyEncoder)

  if args.verbose: print(f'\nSaved stats to: {fn}')

  return stats
