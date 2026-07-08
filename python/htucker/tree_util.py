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