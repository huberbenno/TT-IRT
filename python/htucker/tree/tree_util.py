import numpy as np
import heapq

def linear_tupletree(dims):
  """
  Linear tree over the dimensions in `dims`, in the given order.

  Parameters
  ----------
  dims: sequence of int
    Dimensions at the leaves of the tree.

  Returns
  -------
  tupletree: tuple
    Nested tuples (d_0, (d_1, (..., (d_k,)))), see `Tree.from_tupletree`.
  """
  dims = list(dims)
  tuple = (dims[-1],)
  for i in reversed(dims[:-1]):
    tuple = (i, tuple)

  return tuple

def balanced_binary_tupletree(dims):
  """
  Balanced binary tree over the dimensions in `dims`, in the given order.

  Parameters
  ----------
  dims: sequence of int
    Dimensions at the leaves of the tree.

  Returns
  -------
  tupletree: tuple or int
    Nested tuples describing the tree, see `Tree.from_tupletree`.
    A single int if `dims` has length 1.
  """
  def worker(indices):
    if len(indices) == 1:
      return indices[0]
    else:
      middle = len(indices) // 2
      sub_l = worker(indices[:middle])
      sub_r = worker(indices[middle:])
      return (sub_l, sub_r)
    
  return worker(list(dims))

def invert_tupletree(tupletree):
  """
  Invert the tupletree by inverting the order of the children at each node.
  """
  if isinstance(tupletree, tuple):
    return tuple(invert_tupletree(e) for e in tupletree[::-1])
  else:
    return tupletree

def weighted_binary_tupletree(dims, weights, mode='shannon-fano'):
  """
  Weighted binary tree over the dimensions in `dims`.

  Parameters
  ----------
  dims: sequence of int
    Dimensions at the leaves of the tree.
  weights: array_like
    weights[i] is the weight of dims[i]. Must have the same length as `dims`.
  mode: {'shannon-fano', 'huffmann'}
    'shannon-fano' recursively splits `dims` (in the given order) into halves of
    roughly equal total weight. 'huffmann' builds a Huffman tree, which places
    dimensions with larger weights closer to the root.

  Returns
  -------
  tupletree: tuple or int
    Nested tuples describing the tree, see `Tree.from_tupletree`.
    A single int if `dims` has length 1.
  """
  dims = list(dims)
  assert len(weights) == len(dims), 'dims must match weight vector length'
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

    return worker(dims, weights)

  if mode == 'huffmann':
    class HeapNode:
      def __init__(self, weight, val):
        self.weight = weight
        self.val = val

      def __lt__(self, other):
        return self.weight < other.weight
    
    queue = [HeapNode(weights[i], dims[i]) for i in range(len(dims))]
    heapq.heapify(queue)
    while True:
      try:
        n1 = heapq.heappop(queue)
        n2 = heapq.heappop(queue)
        heapq.heappush(queue, HeapNode(n1.weight+n2.weight, (n1.val,n2.val)))
      except:
        break

    return n1.val

  else:
    raise Exception(f"invalid mode '{mode}'")


def random_tupletree(dims, seed=None):
  """
  Random binary tree over the dimensions in `dims`.

  Parameters
  ----------
  dims: sequence of int
    Dimensions at the leaves of the tree. Not modified.
  seed: int, optional
    Seed for `numpy.random.default_rng`.

  Returns
  -------
  tupletree: tuple or int
    Nested tuples describing the tree, see `Tree.from_tupletree`.
    A single int if `dims` has length 1.
  """
  rng = np.random.default_rng(seed=seed)

  def worker(indices):
    if len(indices) == 1:
      return indices[0]

    split_i = rng.integers(1, len(indices))
    rng.shuffle(indices)
    sub_l = worker(indices[:split_i])
    sub_r = worker(indices[split_i:])
    return (sub_l, sub_r)

  return worker(list(dims))


def zero_root_tupletree(n, tree_type='balanced', **kwargs):
  """
  Tree with dimension 0 as a child of the root and a subtree of type `tree_type`
  over the dimensions 1..n.

  Parameters
  ----------
  n: int
    Number of dimensions in the subtree.
  tree_type: {'linear', 'balanced', 'weighted', 'random'}
    Type of the subtree.
  **kwargs:
    Passed to the builder for `tree_type`, e.g. `weights` and `mode` for 'weighted'
    or `seed` for 'random'.

  Returns
  -------
  tupletree: tuple
    Nested tuples (0, subtree), see `Tree.from_tupletree`.

  Raises
  ------
  ValueError
    If `tree_type` is not one of the types above.
  """
  builders = {
    'linear': linear_tupletree,
    'balanced': balanced_binary_tupletree,
    'weighted': weighted_binary_tupletree,
    'random': random_tupletree,
  }
  if tree_type not in builders:
    raise ValueError(f"invalid tree_type '{tree_type}'")
  return (0, builders[tree_type](list(range(1, n+1)), **kwargs))


def random_nary_tupletree(n, max_arity=3, seed=None):
  """
  Random tree with bounded arity over the dimensions 0..n.

  Parameters
  ----------
  n: int
    The tree has n+1 leaves with dimensions 0..n. Must be at least 1.
  max_arity: int
    Maximum number of children of each inner node. Must be at least 2.
  seed: int, optional
    Seed for `numpy.random.default_rng`.

  Returns
  -------
  tupletree: tuple
    Nested tuples describing the tree, see `Tree.from_tupletree`.
  """
  assert n >= 1, 'n must be at least 1'
  assert max_arity >= 2, 'max_arity must be at least 2'
  rng = np.random.default_rng(seed=seed)

  def worker(indices):
    if len(indices) == 1:
      return indices[0]

    n_children = rng.integers(2, min(max_arity, len(indices)) + 1)
    rng.shuffle(indices)
    splits = [0] + sorted(rng.choice(range(1, len(indices)), size=n_children - 1, replace=False).tolist()) + [len(indices)]
    return tuple(worker(indices[a:b]) for a, b in zip(splits[:-1], splits[1:]))

  return worker(list(range(n + 1)))
