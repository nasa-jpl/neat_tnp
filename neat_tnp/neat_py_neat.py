from neat_tnp.neat_py_core import *
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
from dataclasses import dataclass
from random import random, uniform, gauss
from copy import deepcopy
from collections import deque

class NodeType(Enum):
    Input = auto()
    Hidden = auto()
    Output = auto()

class NodeAct(Enum):
    ReLU = auto()

@dataclass
class NodeData:
    type_: NodeType
    bias: float
    activation: NodeAct

@dataclass
class EdgeData:
    weight: float

class NEATOrigSpecialization(NEATSpecialization):

    def initialize_genome(self, pop: Population) -> Tuple[List[Node], List[Edge]]:
        # fully-connect input to output layer
        conf = self.config.special
        num_output, num_inputs = conf.shape
        input_nodes: List[Node] = []
        output_nodes: List[Node] = []
        for _ in range(num_inputs):
            id = pop.new_node_id()
            input_nodes.append(Node(
                id=id,
                data=NodeData(
                    type_=NodeType.Input,
                    bias=uniform(conf.min_bias, conf.max_bias),
                    activation=NodeAct.ReLU,
                )
            ))
        for _ in range(num_output):
            id = pop.new_node_id()
            output_nodes.append(Node(
                id=id,
                data=NodeData(
                    type_=NodeType.Output,
                    bias=uniform(conf.min_bias, conf.max_bias),
                    activation=NodeAct.ReLU
                )
            ))
        edges: List[Edge] = []
        for node1 in input_nodes:
            for node2 in output_nodes:
                edge_id = pop.get_edge_id(node1.id, node2.id)
                edges.append(Edge(
                    node1=node1.id,
                    node2=node2.id,
                    id=edge_id,
                    data=EdgeData(
                        weight=uniform(conf.min_weight, conf.max_weight)
                    )
                ))
        return input_nodes+output_nodes, edges
    
    def edge_weight_mutation(self, edges: Dict[NodeID, Node]):
        conf = self.config.special
        for edge_id in edges:
            if random() < conf.weight_mutation_prob:
                edge = edges[edge_id]
                edge.data.weight += gauss(0, conf.weight_mutation_sigma)
                edge.data.weight = min(edges[edge_id].data.weight, conf.max_weight)
                edge.data.weight = max(edges[edge_id].data.weight, conf.min_weight)
    
    def node_bias_mutation(self, nodes: Dict[NodeID, Node]):
        conf = self.config.special
        mutable_nodes = [id for id,node in nodes.items() if node.data.type_ != NodeType.Input]
        for node_id in mutable_nodes:
            if random() < conf.bias_mutation_prob:
                node = nodes[node_id]
                node.data.bias += gauss(0, conf.bias_mutation_sigma)
                node.data.bias = min(nodes[node_id].data.bias, conf.max_bias)
                node.data.bias = max(nodes[node_id].data.bias, conf.min_bias)
    
    def mutate(self,
               nodes: Dict[NodeID, Node],
               edges: Dict[EdgeID, Edge],
               ):
        self.edge_weight_mutation(edges)
        self.node_bias_mutation(nodes)
    
    def add_node(self,
                 node: Node,
                 old_edge: Edge,
                 edge_node1_new: Edge,
                 edge_new_node2: Edge,
                 nodes: Dict[NodeID, Node],
                 edges: Dict[EdgeID, Edge],
                 ) -> bool:
        conf = self.config.special

        node.data = NodeData(
            type_=NodeType.Hidden,
            bias=uniform(conf.min_bias, conf.max_bias),
            activation=NodeAct.ReLU,
        )
        edge_node1_new.data = EdgeData(weight=1.0)
        edge_new_node2.data = EdgeData(weight=old_edge.data.weight)

        return True
    
    def remove_node(self, 
                    node: Node, 
                    edge1: Edge,
                    edge2: Edge,
                    new_edge: Edge,
                    nodes: Dict[NodeID, Node],
                    edges: Dict[EdgeID, Edge],
                    ) -> bool:
        
        return node.data.type_ not in (NodeType.Input, NodeType.Output)

    def remove_node_single(self,
                           node: Node,
                           edge: Edge,
                           nodes: Dict[NodeID, Node],
                           edges: Dict[EdgeID, Edge],
                           ) -> bool:
        return node.data.type_ not in (NodeType.Input, NodeType.Output)
    
    def add_edge(self,
                 edge: Edge,
                 nodes: Dict[NodeID, Node],
                 edges: Dict[EdgeID, Edge],
                 ) -> bool:
        conf = self.config.special

        if nodes[edge.node1].data.type_ == NodeType.Output or \
            nodes[edge.node2].data.type_ == NodeType.Input:
            return False
        
        if self.check_cycle(edge, nodes, edges):
            return False
        
        edge.data = EdgeData(
            weight=uniform(conf.min_weight, conf.max_weight),
        )

        return True
    
    def remove_edge(self,
                    edge: 'Edge',
                    nodes: Dict[NodeID, Node],
                    edges: Dict[EdgeID, Edge],
                    ) -> bool:
        if nodes[edge.node1].data.type_ == NodeType.Input and \
            nodes[edge.node2].data.type_ == NodeType.Output:
            return False
        
        return True
    
    # TODO: computing this every time is very inefficient
    def _build_adj_list(self, edges: Dict[EdgeID, Edge]) -> Dict[NodeID, List[NodeID]]:
        """Helper to build an adjacency list on the fly."""
        adj = {}
        for edge in edges.values():
            if not edge.enabled:
                continue
            if edge.node1 not in adj:
                adj[edge.node1] = []
            adj[edge.node1].append(edge.node2)
        return adj
    
    def check_cycle(self, edge: Edge, nodes: Dict[NodeID, Node], edges: Dict[EdgeID, Edge]):
        """
        Check if adding an edge node1 -> node2 would create a cycle.
        i.e., if there currently is a path from node2 to node1.
        """
        # Build the graph on the fly from the *true* data
        adj = self._build_adj_list(edges)

        node1_id, node2_id = edge.node1, edge.node2
        
        stack = [node2_id]
        visited = {node2_id}
        while stack:
            current_node = stack.pop()
            if current_node == node1_id:
                return True # Cycle found!
            if current_node in adj:
                for neighbor_id in adj[current_node]:
                    if neighbor_id in nodes and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        stack.append(neighbor_id)
        return False # No cycle
    
    def _get_topological_sort(self, nodes: Dict[NodeID, Node], edges: Dict[EdgeID, Edge]) -> List[NodeID]:
        """Performs a topological sort (Kahn's algorithm)"""
        adj = {} # out-edges
        in_degree = {node_id: 0 for node_id in nodes}
        
        for edge in edges.values():
            if not edge.enabled:
                continue
            
            # Check if nodes exist before adding
            if edge.node1 not in nodes or edge.node2 not in nodes:
                continue
                
            if edge.node1 not in adj:
                adj[edge.node1] = []
            adj[edge.node1].append(edge.node2)
            
            in_degree[edge.node2] += 1
        
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        processing_order = []
        
        while queue:
            current_node = queue.popleft()
            processing_order.append(current_node)
            
            if current_node in adj:
                for neighbor_node in adj[current_node]:
                    in_degree[neighbor_node] -= 1
                    if in_degree[neighbor_node] == 0:
                        queue.append(neighbor_node)
                            
        return processing_order

    def forward(self, inputs: List[float], nodes: Dict[NodeID, Node], edges: Dict[EdgeID, Edge]) -> List[float]:
        """Computes the forward pass (network activation) for a given set of inputs."""

        # Re-build these on the fly. Caching them is a
        # micro-optimization that adds complexity.
        sorted_input_ids = sorted([
            id for id, node in nodes.items() if node.data.type_ == NodeType.Input
        ])
        sorted_output_ids = sorted([
            id for id, node in nodes.items() if node.data.type_ == NodeType.Output
        ])

        processing_order = self._get_topological_sort(nodes, edges)
        node_values = {node_id: 0.0 for node_id in nodes}
        
        for node_id, value in zip(sorted_input_ids, inputs):
            node_values[node_id] = value
            
        for node_id in processing_order:
            # activate
            if nodes[node_id].data.type_ != NodeType.Input:
                sum_val = node_values[node_id] + nodes[node_id].data.bias
                node_values[node_id] = max(0.0, sum_val) # ReLU
            
            # propagate
            activated_value = node_values[node_id]
            for edge in edges.values():
                if not edge.enabled or edge.node1 != node_id:
                    continue
                
                # Ensure target node exists
                if edge.node2 in node_values:
                    node_values[edge.node2] += activated_value * edge.data.weight

        output_values = [
            node_values[node_id] 
            for node_id in sorted_output_ids
        ]
        return output_values

    def compatibility(self,
                      genome1: Genome, matching1: List[Edge], disjoint1: List[Edge], excess1: List[Edge],
                      genome2: Genome, matching2: List[Edge], disjoint2: List[Edge], excess2: List[Edge]):
        weight_diffs = 0.0
        for e1, e2 in zip(matching1, matching2):
            weight_diffs += abs(e1.data.weight - e2.data.weight)
        return weight_diffs/len(matching1)

@dataclass
class NEATOrigConfig:
    shape: Tuple[int, int]  # network shape in terms of (#output, #inputs)
    weight_mutation_prob: float = 0.8
    weight_mutation_sigma: float = 0.1
    # TODO: the original code also had a reset probability; do we need that?
    min_weight: float = -1.0
    max_weight: float = 1.0
    bias_mutation_prob: float = 0.8
    bias_mutation_sigma: float = 0.1
    # TODO: same here, the original code has a reset probability
    min_bias: float = -1.0
    max_bias: float = 1.0

# TODO: add ablity to modify fitness normalization in neat_py_core, then add weights