"""
Shared type definitions for NEATWork.

This module is the single source of truth for all configuration types,
enums, and the serializable Network format. Both the Rust and Python
reference implementations import from here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict


# Enums

class MatingStrategy(str, Enum):
    NEATSpeciation = "NEATSpeciation"
    GlobalTournament = "GlobalTournament"


class NodeType(str, Enum):
    Fixed = "Fixed"
    Flexible = "Flexible"


class Objective(str, Enum):
    CostMap = "cost_map"
    TrafficBalance = "traffic_balance"
    MaxEdgeTraffic = "max_edge_traffic"


class Constraint(str, Enum):
    NoIntersectingEdges = "no_intersecting_edges"


# Helper classes

class GraphInit:
    FCN = "FCN"

    @staticmethod
    def grid(rows: int, cols: int):
        return {"Grid": (rows, cols)}
        # return f"Grid_{rows}_{cols}"


class ScalarizationStrategy:
    @staticmethod
    def augmented_chebyshev(rho: float = 0.05) -> Dict[str, Any]:
        return {"augmented_chebyshev": {"rho": rho}}


# Network format

class NetworkNode(BaseModel):
    id: int
    pos: Tuple[int, int]
    type: NodeType


class NetworkEdge(BaseModel):
    id: int
    node1: int
    node2: int
    enabled: bool = True


class Network(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    fitness: Optional[float] = None

    def to_dict(self) -> dict:
        """
        Returns the dict shape that Rust ``create_genome`` accepts via serde::

            {"nodes": [{"id": .., "data": {"pos": [r,c], "type": "Fixed"}}, ...],
             "edges": [{"id": .., "node1": .., "node2": .., "enabled": ..}, ...]}
        """
        return {
            "nodes": [
                {"id": n.id, "data": {"pos": list(n.pos), "type": n.type.value}}
                for n in self.nodes
            ],
            "edges": [
                {"id": e.id, "node1": e.node1, "node2": e.node2, "enabled": e.enabled}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Network:
        """Inverse of ``to_dict()``."""
        nodes = [
            NetworkNode(
                id=n["id"],
                pos=tuple(n["data"]["pos"]),
                type=NodeType(n["data"]["type"]),
            )
            for n in d["nodes"]
        ]
        edges = [
            NetworkEdge(
                id=e["id"],
                node1=e["node1"],
                node2=e["node2"],
                enabled=e.get("enabled", True),
            )
            for e in d["edges"]
        ]
        return cls(nodes=nodes, edges=edges)

    @property
    def fixed_nodes(self) -> List[NetworkNode]:
        return [n for n in self.nodes if n.type == NodeType.Fixed]

    @property
    def dynamic_nodes(self) -> List[NetworkNode]:
        return [n for n in self.nodes if n.type == NodeType.Flexible]


# Configs

class NWConfig(BaseModel):
    grid_size: Tuple[int, int] = (10, 10)
    fixed_nodes: List[Tuple[int, int]] = []
    move_node_mutation_prob: float = 0.6
    move_node_mutation_sigma: float = 5.0
    graph_init: Any = GraphInit.FCN
    compatibility_sigma: int = 5
    c3: float = 1.0


class EvalConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cost_map: Any  # np.ndarray — Any to avoid pydantic numpy validation issues
    objectives: List[Objective] = []
    constraints: List[Constraint] = []
    scalarization: Optional[Dict[str, Any]] = None


class NEATCoreConfig(BaseModel):
    # Population
    population_size: int = 100

    # Mutation probabilities
    add_node_mutation_prob: float = 0.4
    add_edge_mutation_prob: float = 0.4
    remove_edge_mutation_prob: float = 0.1
    remove_node_mutation_prob: float = 0.1
    remove_node_single_mutation_prob: float = 0.1

    # Reproduction & selection
    num_elites_species: int = 1
    num_elites_global: int = 1
    selection_share: float = 0.2
    tournament_size: int = 2
    mating_strategy: MatingStrategy = MatingStrategy.NEATSpeciation
    crossover_enabled: bool = True

    # Speciation
    species_threshold: float = 3.0
    target_species: Optional[int] = None
    precompute_threshold: bool = False

    # Compatibility distance coefficients
    c1: float = 1.0
    c2: float = 1.0

    # Specialization config
    special: Any = None

    # Run parameters
    ngen: Optional[int] = None
    target_fit: Optional[float] = None
    start_genome: Optional[Network] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
