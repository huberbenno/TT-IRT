import pydot
from tree.tree import Tree

def tree_to_graph(tree: Tree) -> pydot.Dot:
  graph = pydot.Dot("tree", graph_type='graph')

  for node in tree.node_list:
    if node.isleaf:
      graph.add_node(pydot.Node(f'{node.id}', label=f'{node.dim}', color='blue'))
    else:
      graph.add_node(pydot.Node(f'{node.id}', label=''))
    if not node.isroot:
      graph.add_edge(pydot.Edge(f'{node.parent.id}', f'{node.id}'))

  return graph