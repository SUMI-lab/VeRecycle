import networkx as nx
from typing import Tuple, List

import numpy as np
from networkx import cytoscape_data, cytoscape_graph

from core.commons import RectangularSet, MultiRectangularSet


class TaskDecompositionGraph(nx.DiGraph):

    def __init__(self, nodes: List[Tuple[int, List[Tuple[List[float], List[float]]]]], edges: List[Tuple[int, int]],
                 initial_node: int, target_node: int) -> None:
        super().__init__(initial_node=initial_node, target_node=target_node)
        for node in nodes:
            self.add_node(node[0], region=node[1])
        for edge in edges:
            self.add_edge(edge[0], edge[1])
        return

    @classmethod
    def from_dict(cls, dictionary):
        digraph = cytoscape_graph(dictionary)
        nodes = [(node, data.get('region', [])) for node, data in digraph.nodes(data=True)]
        edges = list(digraph.edges)
        initial_node = digraph.graph.get('initial_node', None)
        target_node = digraph.graph.get('target_node', None)

        return cls(nodes, edges, initial_node, target_node)

    def to_dict(self):
        return cytoscape_data(self)

    def get_region(self, node_index: int) -> MultiRectangularSet:
        sets = []
        for s in self.nodes[node_index]['region']:
            sets.append(RectangularSet(low=np.array(s[0]), high=np.array(s[1])))
        return MultiRectangularSet(sets)

    def get_initial_node(self) -> int:
        return self.graph['initial_node']

    def get_target_node(self) -> int:
        return self.graph['target_node']

    def get_initial_region(self) -> MultiRectangularSet:
        return self.get_region(self.get_initial_node())

    def get_target_region(self) -> MultiRectangularSet:
        return self.get_region(self.get_target_node())
