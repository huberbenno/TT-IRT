import copy
import pytest

from tree.tree import Tree, TreeNode, NodeIndexedList
from tree.tree_util import zero_root_tupletree, random_nary_tupletree


TREES = {
  'balanced': zero_root_tupletree(7, 'balanced'),
  'linear': zero_root_tupletree(5, 'linear'),
  'random': zero_root_tupletree(7, 'random', seed=1),
  'nary': random_nary_tupletree(6, max_arity=4, seed=5),
  'example': (0, (1, 2, 3), (4, 5)),
  'single_leaf': 0,
}

def make(tree_type):
  return Tree.from_tupletree(TREES[tree_type])

def follow(tree, path):
  """ Node reached from the root by following a list of child indices. """
  node = tree.root
  for i in path:
    node = node.children[i]
  return node


# ---------- construction from tupletrees ----------

def test_example_structure():
  # nodes are numbered in post-order: children before their parent
  tree = make('example')
  assert tree.n_nodes == 9
  assert tree.order == 6
  assert tree.arity == 3
  assert tree.root.id == 8
  assert [c.id for c in tree.root.children] == [0, 4, 7]
  assert [c.id for c in tree.node_list[4].children] == [1, 2, 3]
  assert [tree.dim2id(d) for d in range(6)] == [0, 1, 2, 3, 5, 6]

def test_single_leaf():
  tree = make('single_leaf')
  assert tree.n_nodes == 1 and tree.order == 1 and tree.arity == 0
  assert tree.root.isroot and tree.root.isleaf
  assert tree.dim2leaf(0) is tree.root

@pytest.mark.parametrize('tupletree', [(0, 2), (1, 2), (0, (1, 1))])
def test_invalid_dims(tupletree):
  with pytest.raises(AssertionError):
    Tree.from_tupletree(tupletree)

@pytest.mark.parametrize('tupletree', [(0, 1.), (0, '1'), (0, [1, 2])])
def test_invalid_types(tupletree):
  with pytest.raises(TypeError):
    Tree.from_tupletree(tupletree)

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_counts(tree_type):
  tree = make(tree_type)
  leafs = [node for node in tree.node_list if node.isleaf]
  assert tree.order == len(leafs)
  assert tree.n_nodes == len(tree.node_list)
  assert [node.id for node in tree.node_list] == list(range(tree.n_nodes))
  assert tree.arity == max(node.n_children for node in tree.node_list)


# ---------- node relations ----------

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_root(tree_type):
  tree = make(tree_type)
  roots = [node for node in tree.node_list if node.isroot]
  assert roots == [tree.root]
  assert tree.root.parent is None
  assert tree.root.siblings is None
  assert tree.root.child_ind is None

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_parent_child_consistency(tree_type):
  tree = make(tree_type)
  for node in tree.node_list:
    assert node.tree is tree
    assert node.n_children == len(node.children)
    assert node.isleaf == (node.n_children == 0)
    assert (node.dim is not None) == node.isleaf
    for i, child in enumerate(node.children):
      assert child.parent is node
      assert child.child_ind == i
    if not node.isroot:
      assert node.parent.children[node.child_ind] is node

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_siblings(tree_type):
  tree = make(tree_type)
  for node in tree.node_list:
    if node.isroot: continue
    expected = [c for c in node.parent.children if c.id != node.id]
    assert [s.id for s in node.siblings] == [s.id for s in expected]
    # computing siblings must not change the tree
    assert node in node.parent.children

def test_children_setter():
  node = TreeNode(id=0)
  node.children = (1, 2)
  assert node._children == (1, 2)
  node.children = (TreeNode(id=3), TreeNode(id=4))
  assert node._children == (3, 4)
  with pytest.raises(AssertionError):
    node.children = [1, 2]
  with pytest.raises(TypeError):
    node.children = (1, TreeNode(id=2))


# ---------- dims, depths, subtrees, paths ----------

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_dim2leaf(tree_type):
  tree = make(tree_type)
  for d in range(tree.order):
    leaf = tree.dim2leaf(d)
    assert leaf.isleaf and leaf.dim == d
    assert tree.dim2id(d) == leaf.id

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_depths(tree_type):
  tree = make(tree_type)
  depths = tree.get_depths()
  levels = tree.get_levels()
  for node in tree.node_list:
    expected = 0 if node.isroot else depths[node.parent] + 1
    assert depths[node] == expected
    assert levels[node.id] == expected
  assert tree.get_leaf_depths() == [depths[tree.dim2leaf(d)] for d in range(tree.order)]

def test_example_depths():
  tree = make('example')
  assert tree.get_leaf_depths() == [1, 2, 2, 2, 2, 2]

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_subtree_dims(tree_type):
  tree = make(tree_type)
  subtree_dims = tree.get_subtree_dims()
  assert subtree_dims[tree.root] == tuple(range(tree.order))
  for node in tree.node_list:
    if node.isleaf:
      assert subtree_dims[node] == (node.dim,)
    else:
      assert subtree_dims[node] == tuple(sorted(d for c in node.children for d in subtree_dims[c]))

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_paths(tree_type):
  tree = make(tree_type)
  depths = tree.get_depths()
  assert tree.path_to_node(tree.root) == []
  for node in tree.node_list:
    path = tree.path_to_node(node)
    assert len(path) == depths[node]
    assert follow(tree, path) is node
  for d in range(tree.order):
    assert follow(tree, tree.path_to_dim(d)) is tree.dim2leaf(d)

def test_example_paths():
  tree = make('example')
  assert tree.path_to_dim(0) == [0]
  assert tree.path_to_dim(3) == [1, 2]
  assert tree.path_to_dim(5) == [2, 1]


# ---------- equality and copies ----------

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_equal(tree_type):
  tree = make(tree_type)
  assert tree == make(tree_type)
  assert tree == copy.deepcopy(tree)

def test_not_equal():
  trees = {name: make(name) for name in TREES}
  for name1, tree1 in trees.items():
    for name2, tree2 in trees.items():
      if name1 != name2:
        assert tree1 != tree2, f'{name1} == {name2}'
  assert make('example') != 'not a tree'

def test_not_equal_prefix():
  # the node list of the first tree is a prefix of the second's
  assert Tree.from_tupletree((0, 1)) != Tree.from_tupletree(((0, 1), 2))

def test_node_not_equal_more_children():
  node1 = TreeNode(id=0, _children=(1, 2))
  node2 = TreeNode(id=0, _children=(1, 2, 3))
  assert node1 != node2

def test_deepcopy_independent():
  tree = make('example')
  tree_copy = copy.deepcopy(tree)
  assert all(node.tree is tree_copy for node in tree_copy.node_list)
  tree_copy.root.children = (0, 4)
  assert tree.root.n_children == 3


# ---------- NodeIndexedList and printing ----------

def test_node_indexed_list():
  tree = make('example')
  values = NodeIndexedList(range(tree.n_nodes))
  for node in tree.node_list:
    assert values[node] == node.id == values[node.id]
  values[tree.root] = -1
  assert values[tree.root.id] == -1
  assert values[1:3] == [1, 2]

@pytest.mark.parametrize('tree_type', TREES.keys())
def test_repr_print(tree_type, capsys):
  tree = make(tree_type)
  assert str(tree.order) in repr(tree)
  for node in tree.node_list:
    assert f'[{node.id}]' in repr(node)
  tree.print()
  out = capsys.readouterr().out
  for node in tree.node_list:
    assert f'[{node.id}]' in out
