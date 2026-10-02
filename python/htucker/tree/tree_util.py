import numpy as np
import heapq

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

def weighted_binary_tupletree(n, weights, mode='shannon-fano'):
  assert len(weights) == n, 'n must match weight vector length'
  if mode == 'shannon-fano':
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

  if mode == 'huffmann':
    class HeapNode:
      def __init__(self, weight, val):
        self.weight = weight
        self.val = val

      def __lt__(self, other):
        return self.weight < other.weight
    
    queue = [HeapNode(weights[i], i+1) for i in range(0,n)]
    heapq.heapify(queue)
    while True:
      try:
        n1 = heapq.heappop(queue)
        n2 = heapq.heappop(queue)
        heapq.heappush(queue, HeapNode(n1.weight+n2.weight, (n1.val,n2.val)))
      except:
        break

    return (0, n1.val)

  else:
    raise Exception(f"invalid mode '{mode}'")


def random_tupletree(n, seed=None):
  rng = np.random.default_rng(seed=seed)

  def worker(indices):
    if len(indices) == 1:
      return indices[0]

    split_i = rng.integers(1, len(indices))
    rng.shuffle(indices)
    sub_l = worker(indices[:split_i])
    sub_r = worker(indices[split_i:])
    return (sub_l, sub_r)

  return (0, worker([i+1 for i in range(n)]))
