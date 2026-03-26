from tree_tensor import TreeBasedTensor
from tree_cross import TreeCross
import numpy as np

from tree import Tree

print('###### Tree tensor init')
tree = Tree.from_tupletree((0,(1,((2,3), (4,5,6)))))
shape = (100,9,8,7,6,5,4)
ttensor = TreeBasedTensor.randn(tree, shape, min(shape))
ttensor.print()

def target_f(inds):
  # return  (1 + np.sum(inds, axis=-1)**2)
  inds = (inds / np.array(shape)) * (1+np.arange(ttensor.ndim))**-2.
  inds = np.moveaxis(inds, -1,0)
  return  1+np.exp(np.sum(inds,axis=0))

inds = np.meshgrid(*[np.arange(d, dtype=int) for d in ttensor.shape], indexing='ij')
inds = np.stack(inds, axis=-1)

target_full = target_f(inds)

print('###### Running cross')
cross = TreeCross(ttensor)
cross.run(target_f, n_iter=10, eps=1e-9, kickrank=2, verbose=True)
ttensor.print()
ht_full = ttensor.numpy()
print(f'err = {np.linalg.norm(target_full - ht_full) / np.linalg.norm(target_full):.2e}')

tol = 1e-5
print(f'###### Rounding with tol={tol:.2e}')
ttensor.round(tol=tol)
ttensor.print()
ht_full = ttensor.numpy()
print(f'err = {np.linalg.norm(target_full - ht_full) / np.linalg.norm(target_full):.2e}')
# print(target_full)
# print(ht_full)
