from htcross import HTCrossTreeNode, HTCrossTree
import numpy as np

tree = HTCrossTree()

# tree.root = HTCrossTreeNode(
#   [
#     HTCrossTreeNode(
#       [
#         HTCrossTreeNode(None, 2, 10, 0),
#         HTCrossTreeNode(None, 2, 10, 1),
#       ],
#       r=2
#     ), 
#     HTCrossTreeNode(
#       [
#         HTCrossTreeNode(None, 2, 10, 2),
#         HTCrossTreeNode(None, 2, 10, 3),
#       ],
#       r=2
#     ),    
#   ],
#   r=1
# )
tree.root = HTCrossTreeNode(
    [
      HTCrossTreeNode(None, 2, 3, 0),
      HTCrossTreeNode(
        [
          HTCrossTreeNode(None, 2, 3, 1),
          HTCrossTreeNode(
            [
              HTCrossTreeNode(None, 2, 3, 2),
              HTCrossTreeNode(
                [
                  HTCrossTreeNode(None, 2, 3, 3),
                  HTCrossTreeNode(None, 2, 3, 4),
                ],
                r=2)
            ],
            r=2)
        ],
        r=2)
    ],
    r=1
  )
tree.setup()
tree.init_randn()
print(tree.shape)

def target_f(inds):
  # return (1 + np.sum(inds, axis=-1))
  # return (1 + np.sum(np.arange(1,inds.shape[-1]+1)**-4. * inds, axis=-1))
  # inds = np.moveaxis(inds, -1, 0)
  return 1 + inds.T[2]

inds = np.meshgrid(*[np.arange(d, dtype=int) for d in tree.shape], indexing='ij')
inds = np.stack(inds, axis=-1)

target_full = target_f(inds)

ht_full = tree.to_full()
print(f'err = {np.linalg.norm(target_full - ht_full) / np.linalg.norm(target_full):.2e}')

# for 
tree.run_cross(target_f, iterations=6, eps=0)

ht_full = tree.to_full()

print(f'err = {np.linalg.norm(target_full - ht_full)/ np.linalg.norm(target_full):.2e}')

print(tree)

