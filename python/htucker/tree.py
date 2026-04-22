from dataclasses import dataclass
from collections import deque
import numpy as np

class NodeIndexedList(list):
  """
  Simple wrapper around list which can be indexed using tree nodes.
  """
  def __init__(self, *args):
    super().__init__(*args)

  def __getitem__(self, ind):
    if isinstance(ind, TreeNode):
      return super().__getitem__(ind.id)
    else:
      return super().__getitem__(ind)

  def __setitem__(self, ind, val):
    if isinstance(ind, TreeNode):
      super().__setitem__(ind.id, val)
    else:
      super().__setitem__(ind, val)

@dataclass
class TreeNode:
  id : int = None
  _children : tuple[int] = tuple()
  _parent : int = None
  tree : Tree = None
  dim : int = None

  @property
  def children(self) -> list[TreeNode]:
    """
    List of the nodes children. None for leafs.
    """
    return [self.tree.node_list[c] for c in self._children]
  
  @children.setter
  def children(self, children):
    assert isinstance(children, tuple), 'children must be given as tuple.'
    if all(isinstance(c, int) for c in children):
      self._children = children
    elif all(isinstance(c, TreeNode) for c in children):
      self._children = tuple(c.id for c in children)
    else:
      raise TypeError('children must be tuple of int or TreeNode.')

  @property
  def n_children(self) -> int:
    """
    Number of children of the node.
    """
    return len(self._children)
  
  @property
  def parent(self) -> TreeNode | None:
    """
    Parent of the node. None for root.
    """
    if self.isroot:
      return None
    else:
      return self.tree.node_list[self._parent]
    
  @parent.setter
  def parent(self, parent):
    self._parent = parent.id

  @property
  def siblings(self) -> list[TreeNode] | None:
    """
    Siblings of the node, i.e., the other children of the parent. None for root.
    """
    if self.isroot:
      return None
    else:
      siblings = self.parent.children
      siblings.remove(self)
      return siblings
  
  @property
  def child_ind(self) -> int:
    if self.isroot():
      return None
    else:
      return self.parent.children.index(self)

  @property
  def isleaf(self) -> bool:
    return len(self.children) == 0
  
  @property
  def isroot(self) -> bool:
    return self._parent is None

  def __eq__(self, other) -> bool:
    return isinstance(other, TreeNode) \
      and self.id == other.id \
      and self._parent == other._parent \
      and self.dim == other.dim \
      and all(c1 == c2 for c1,c2 in zip(self._children, other._children))
    
  
  def __repr__(self) -> str:
    if self.isroot:
      return f'Node [{self.id}] (root) with children {tuple(child.id for child in self.children)}'
    elif self.isleaf:
      return f'Node [{self.id}] (leaf) for dim {self.dim}'
    else:
      return f'Node [{self.id}] (interior) with parent {self.parent.id} and children {tuple(child.id for child in self.children)}'


class Tree:
  def __init__(self):
    self.node_list = []
    self._root = None
    self._order = 0
    self._dim2id = np.zeros(0, dtype=int)

  @property
  def n_nodes(self) -> int:
    """
    Number of nodes in the tree.
    """
    return len(self.node_list)
  
  @property
  def order(self) -> int:
    """
    Number of leafs in the tree.
    """
    return self._order
  
  @property
  def root(self) -> TreeNode:
    """
    Root node of the tree
    """
    return self.node_list[self._root]
  
  @property
  def arity(self) -> int:
    """
    Maximum number of children of any node in the tree.
    """
    return max(n.n_children for n in self.node_list)
  
  def __eq__(self, other):
    
    return isinstance(other, Tree) and all(n1 == n2 for n1,n2 in zip(self.node_list, other.node_list))
  
  def dim2id(self, dim: int) -> int:
    """
    Map dim to the id of the corresponding leaf.
    """
    return self._dim2id[dim]
  
  def dim2leaf(self, dim: int) -> TreeNode:
    """
    Map dim to the corresponding leaf.
    """
    return self.node_list[self.dim2id(dim)]
  
  def get_levels(self) -> np.ndarray:
    """
    Compute the level of each node, starting from the root with level 0.

    Returns
    -------
    levels: numpy.ndarray
      levels[id] is the level of the node node assciated with id.
    """
    level = 0
    levels = np.zeros(self.n_nodes, dtype=int)
    queue = [(self.root, 0)]
    
    try:
      while True:
        node, level = queue.pop()
        levels[node.id] = level
        for child in node.children:
          queue.append((child, level+1))
    except IndexError:      
      return levels
    
  def get_subtree_dims(self) -> list:
    subtree_dims = self.n_nodes * [None]
    def worker(node):
      if node.isleaf:
        subtree_dims[node.id] = (node.dim, )
      else:
        local_dims = []
        for child in node.children:
          worker(child)
          local_dims += list(subtree_dims[child.id])

        subtree_dims[node.id] = tuple(sorted(local_dims))

    worker(self.root)

    return subtree_dims
  
  def path_to_node(self, node: TreeNode) -> list:
    """
    Path from root to node, given via list of child indices.
    """
    path = []
    while not node.isroot:
      path.append(node.child_ind)
      node = node.parent
    
    return path.reverse()
  
  def path_to_dim(self, dim: int) -> list:
    """
    Path from root to node associated with dim, given via list of child indices.
    """
    node = self.dim2leaf(dim)
    return self.path_to_node(node)

  @staticmethod
  def from_tupletree(tupletree):
    """
    Construct a tree from nested tuples. 
    Integer values mark leaves correspondingto the dimension given by the integer. 
    Dimensions must be unique.

    Parameters
    ----------
    tupletree:
      Nested tuples describing the tree structure. Elements of each tuple correspond
      to children.

    """
    new_tree = Tree()

    leafs = []

    # recurively build tree from subtrees
    def recursive_builder(ttree):
      if isinstance(ttree, tuple):
        children = tuple(recursive_builder(subtree) for subtree in ttree)
        new_node = TreeNode(id=new_tree.n_nodes, tree=new_tree)
        new_node.children = children
        # set parent for the children
        for child in new_node.children:
          child.parent = new_node

      elif isinstance(ttree, int):
        new_node = TreeNode(id=new_tree.n_nodes, dim = ttree, tree=new_tree)
        new_tree._order += 1
        leafs.append(new_node)
        
      else:
        raise TypeError(f'Input must be tuple or int not {type(tupletree)}')
      
      new_tree.node_list.append(new_node)
      return new_node.id
    
    new_tree._root = recursive_builder(tupletree)
    leafs = sorted(leafs, key=lambda node: node.dim)
    assert all(i==j for i,j in zip(tuple(n.dim for n in leafs), range(len(leafs)))), \
      'Invalid dimensions.'

    new_tree._dim2id = np.array([n.id for n in leafs])

    return new_tree
    
  def __repr__(self):
    return f'Tree with  order: {self.order},  nodes: {self.n_nodes},  arity: {self.arity}'
  
  def print(self):
    """
    Print a representation of the tree.
    """
    print(self._print(self.root))

  def _print(self, node, prefix='', last=False):
    if node.isleaf:
      return prefix[3:] + f' + [{node.id}] (dim {node.dim})\n'
    else:
      if node.isroot:
        str = f'[{node.id}] (root) \n'
      else:
        str = prefix[3:] + f' + [{node.id}] \n'
      
      prefix += '   ' if last else ' | '
      for child in node.children[:-1]:
        str += self._print(child, prefix=prefix)
      str += self._print(node.children[-1], prefix=prefix, last=True)
      return str

    