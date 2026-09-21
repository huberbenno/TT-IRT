import numpy as np

def linear_tupletree(n):
  tuple = (n,)
  for i in reversed(range(n)):
    tuple = (i, tuple)

  return tuple

def balanced_binary_tupletree(n):
  def worker(indices):
    if len(indices) == 1:
      return indices[0]
    else:
      middle = len(indices) // 2
      sub_l = worker(indices[:middle])
      sub_r = worker(indices[middle:])
      return (sub_l, sub_r)
    
  return (0, worker([i+1 for i in range(n)]))

def invert_tupletree(tupletree):
  if isinstance(tupletree, tuple):
    return tuple(invert_tupletree(e) for e in tupletree[::-1])
  else:
    return tupletree

def weighted_binary_tupletree(n, weights):
  assert len(weights) == n, 'n must match weight vector length'

  def worker(indices, weights):
    if len(indices) == 1:
      return indices[0]
    cs = np.cumsum(weights)
    split_i = np.argmin(np.abs(cs - .5 * cs[-1])) + 1
    split_i = min(split_i, len(weights))
    sub_l = worker(indices[:split_i], weights[:split_i])
    sub_r = worker(indices[split_i:], weights[split_i:])
    return (sub_l, sub_r)

  return (0, worker([i+1 for i in range(n)], weights))
