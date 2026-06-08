# A specialization of NEAT-core for Track Network Planning (NEATWork)

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from random import random, choice, gauss, randint
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from neatwork.neat_core_py import *
import neatwork.neat_core_py as neat_core_py
import neatwork.neatwork_py_eval as neatwork_py_eval

from neatwork.types import *


@dataclass
class NodeData:
    type_: NodeType
    pos: Tuple[int, int]

class Grid:
    def __init__(self, size: Tuple[int, int]):
        self.h = int(size[0])
        self.w = int(size[1])

    def clip(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        return (
            max(0, min(self.h - 1, pos[0])),
            max(0, min(self.w - 1, pos[1])),
        )

    def is_occupied(self, pos: Tuple[int, int], nodes: Dict['NodeID', 'Node'], ignore: Optional[int] = None) -> bool:
        for node_id, node in nodes.items():
            if ignore is not None and node_id == ignore:
                continue
            if _get_node_pos(node) == pos:
                return True
        return False

    def find_free_pos(self, center: Tuple[int, int], nodes: Dict['NodeID', 'Node']) -> Optional[Tuple[int, int]]:
        if not self.is_occupied(center, nodes):
            return center
        max_rad = max(self.h, self.w)
        for r in range(1, max_rad + 1):
            for _ in range(10):
                dr = randint(-r, r)
                dc = randint(-r, r)
                pos = self.clip((center[0] + dr, center[1] + dc))
                if not self.is_occupied(pos, nodes):
                    return pos
        return None

def _get_node_pos(node: 'Node') -> Tuple[int, int]:
    data = node.data
    if hasattr(data, "pos"):
        return data.pos
    return tuple(data.get("pos"))

def _get_node_type(node: 'Node') -> NodeType:
    data = node.data
    if hasattr(data, "type_"):
        type_val = data.type_
    else:
        type_val = data.get("type") or data.get("type_")
    if isinstance(type_val, NodeType):
        return type_val
    if isinstance(type_val, str):
        return NodeType.Fixed if type_val.lower() == "fixed" else NodeType.Flexible
    return NodeType.Fixed if type_val == 0 else NodeType.Flexible

class NEATWork(NEATSpecialization):

    def _init_fixed_nodes(self, pop: Population) -> List[Node]:
        return [
            Node(
                id=pop.new_node_id(),
                data=NodeData(type_=NodeType.Fixed, pos=p),
            )
            for p in self.config.special.fixed_nodes
        ]

    def _init_fcn_edges(self, pop: Population, nodes: List[Node]) -> List[Edge]:
        edges: List[Edge] = []
        for i1, n1 in enumerate(nodes):
            for n2 in nodes[i1 + 1:]:
                assert n1.id < n2.id
                edges.append(Edge(
                    node1=n1.id,
                    node2=n2.id,
                    id=pop.get_edge_id(n1.id, n2.id),
                ))
        return edges

    def _init_grid_nodes(self, pop: Population, grid: Grid, rows: int, cols: int, nodes: List[Node]) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, Tuple[int, int]]]]:
        grid_ids: Dict[Tuple[int, int], int] = {}
        flex_pos: List[Tuple[int, Tuple[int, int]]] = []
        node_map: Dict[int, Node] = {n.id: n for n in nodes}

        step_r = grid.h / float(rows)
        step_c = grid.w / float(cols)

        for r in range(rows):
            for c in range(cols):
                ideal = (
                    int(round((r + 0.5) * step_r)),
                    int(round((c + 0.5) * step_c)),
                )
                pos = ideal if not grid.is_occupied(ideal, node_map) else grid.find_free_pos(ideal, node_map)
                if pos is None:
                    continue
                node_id = pop.new_node_id()
                node = Node(id=node_id, data=NodeData(type_=NodeType.Flexible, pos=pos))
                nodes.append(node)
                node_map[node_id] = node
                grid_ids[(r, c)] = node_id
                flex_pos.append((node_id, pos))

        return grid_ids, flex_pos

    def _init_grid_edges(self, pop: Population, grid_ids: Dict[Tuple[int, int], int], rows: int, cols: int) -> List[Edge]:
        edges: List[Edge] = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in grid_ids:
                    u = grid_ids[(r, c)]
                    for dr, dc in ((0, 1), (1, 0)):
                        if (r + dr, c + dc) in grid_ids:
                            v = grid_ids[(r + dr, c + dc)]
                            n1, n2 = self.normalize_edge(u, v)
                            edges.append(Edge(
                                node1=n1,
                                node2=n2,
                                id=pop.get_edge_id(n1, n2),
                            ))
        return edges

    def _connect_fixed_to_nearest(self, pop: Population, nodes: List[Node], flex_pos: List[Tuple[int, Tuple[int, int]]]) -> List[Edge]:
        edges: List[Edge] = []
        for fixed in [n for n in nodes if _get_node_type(n) == NodeType.Fixed]:
            if not flex_pos:
                break
            best_id, _ = min(
                flex_pos,
                key=lambda fp: (fixed.data.pos[0] - fp[1][0]) ** 2 + (fixed.data.pos[1] - fp[1][1]) ** 2,
            )
            n1, n2 = self.normalize_edge(fixed.id, best_id)
            edges.append(Edge(
                node1=n1,
                node2=n2,
                id=pop.get_edge_id(n1, n2),
            ))
        return edges

    def initialize_genome(self, pop: Population) -> Tuple[List[Node], List[Edge]]:
        nodes = self._init_fixed_nodes(pop)
        graph_init = self.config.special.graph_init

        if graph_init == GraphInit.FCN or graph_init == "FCN":
            edges = self._init_fcn_edges(pop, nodes)
            return nodes, edges

        if not (isinstance(graph_init, dict) and "Grid" in graph_init):
            raise ValueError(f"Unsupported graph_init: {graph_init}")

        rows, cols = graph_init["Grid"]
        grid = Grid(self.config.special.grid_size)
        grid_ids, flex_pos = self._init_grid_nodes(pop, grid, rows, cols, nodes)
        edges = self._init_grid_edges(pop, grid_ids, rows, cols)
        edges.extend(self._connect_fixed_to_nearest(pop, nodes, flex_pos))
        return nodes, edges

    def add_node(self, node, old_edge, edge_node1_new, edge_new_node2, nodes, edges) -> bool:
        p1 = _get_node_pos(nodes[old_edge.node1])
        p2 = _get_node_pos(nodes[old_edge.node2])
        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        grid = Grid(self.config.special.grid_size)
        new_pos = grid.find_free_pos(midpoint, nodes)
        if new_pos is None:
            return False
        node.data = NodeData(type_=NodeType.Flexible, pos=new_pos)
        return True

    def remove_node(self, node, edge1, edge2, new_edge, nodes, edges) -> bool:
        if _get_node_type(node) != NodeType.Flexible:
            return False
        remaining_node_ids = [n_id for n_id in nodes if n_id != node.id]
        remaining_edges_data = {
            eid: (e.node1, e.node2)
            for eid, e in edges.items()
            if e.node1 != node.id and e.node2 != node.id and e.enabled
        }
        return check_connectivity(remaining_node_ids, remaining_edges_data)

    def remove_node_single(self, node, edge, nodes, edges) -> bool:
        if _get_node_type(node) != NodeType.Flexible:
            return False
        remaining_node_ids = [n_id for n_id in nodes if n_id != node.id]
        remaining_edges_data = {
            eid: (e.node1, e.node2)
            for eid, e in edges.items()
            if e.node1 != node.id and e.node2 != node.id and e.enabled
        }
        return check_connectivity(remaining_node_ids, remaining_edges_data)

    def add_edge(self, edge, nodes, edges) -> bool:
        return True

    def remove_edge(self, edge, nodes, edges) -> bool:
        all_node_ids = list(nodes.keys())
        remaining_edges_data = {
            eid: (e.node1, e.node2)
            for eid, e in edges.items()
            if eid != edge.id and e.enabled
        }
        return check_connectivity(all_node_ids, remaining_edges_data)

    def move_node_mutation(self, nodes: Dict[NodeID, Node]):
        conf = self.config.special
        if random() < conf.move_node_mutation_prob and nodes:
            movable = [id for id, n in nodes.items() if _get_node_type(n) == NodeType.Flexible]
            if not movable:
                return
            node_id = choice(movable)
            x1, x2 = _get_node_pos(nodes[node_id])
            x1 = int(round(gauss(x1, sigma=conf.move_node_mutation_sigma)))
            x2 = int(round(gauss(x2, sigma=conf.move_node_mutation_sigma)))
            grid = Grid(conf.grid_size)
            new_pos = grid.clip((x1, x2))
            if not grid.is_occupied(new_pos, nodes, ignore=node_id):
                nodes[node_id].data.pos = new_pos

    def mutate(self, nodes, edges):
        self.move_node_mutation(nodes)

    @staticmethod
    def normalize_edge(node_id1: NodeID, node_id2: NodeID) -> Tuple[NodeID, NodeID]:
        return (min(node_id1, node_id2), max(node_id1, node_id2))

    def compatibility(self,
                      genome1, matching1, disjoint1, excess1,
                      genome2, matching2, disjoint2, excess2):
        conf = self.config.special
        h, w = conf.grid_size
        t1 = neatwork_py_eval.compute_traces(h, w, genome1)
        t2 = neatwork_py_eval.compute_traces(h, w, genome2)
        s1 = max_pool(t1, conf.compatibility_sigma)
        s2 = max_pool(t2, conf.compatibility_sigma)
        return np.sqrt(np.sum((s1 - s2) ** 2)) * conf.c3

def max_pool(input_array: np.ndarray, block: int) -> np.ndarray:
    h, w = input_array.shape
    nh, nw = h // block, w // block
    out = np.zeros((nh, nw), dtype=np.float64)
    for r in range(nh):
        for c in range(nw):
            out[r, c] = input_array[r*block:(r+1)*block, c*block:(c+1)*block].max()
    return out

def check_connectivity(nodes: List[int], edges: Dict[int, Tuple[int, int]]) -> bool:
    if not nodes:
        return True
    adj = {node_id: [] for node_id in nodes}
    for (n1, n2) in edges.values():
        adj[n1].append(n2)
        adj[n2].append(n1)
    visited = set()
    stack = [next(iter(nodes))]
    while stack:
        cur = stack.pop()
        if cur not in visited:
            visited.add(cur)
            stack.extend(n for n in adj[cur] if n not in visited)
    return len(visited) == len(nodes)

def get_concurrent_executor(num_cores: int = 1):
    class SerialExecutor:
        def __init__(self):
            self._max_workers = 1
        def map(self, fn, iterable):
            return map(fn, iterable)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    if num_cores > 1:
        return ProcessPoolExecutor(max_workers=num_cores)
    return SerialExecutor()

def concurrent_eval_builder(executor, f_eval_batch: Callable[[List[Genome]], List[float]]):
    def f_eval_parallel(genomes: List[Genome]):
        if not genomes:
            return []
        pop = genomes[0].population
        for g in genomes:
            g.population = None
        m = genomes
        n = executor._max_workers
        k, r = divmod(len(m), n)
        batches = []
        start = 0
        for i in range(n):
            end = start + k + (1 if i < r else 0)
            batches.append(m[start:end])
            start = end
        fits = list(executor.map(f_eval_batch, batches))
        fitnesses = [item for f in fits if f is not None for item in f]
        for g in genomes:
            g.population = pop
        return fitnesses
    return f_eval_parallel

def get_genome_mask(genome: Genome, shape: Tuple[int, int]) -> np.ndarray:
    return neatwork_py_eval.compute_traces(shape[0], shape[1], genome)

def get_trace_difference(g1: Genome, g2: Genome, shape: Tuple[int, int], sigma: int) -> np.ndarray:
    t1 = neatwork_py_eval.compute_traces(shape[0], shape[1], g1)
    t2 = neatwork_py_eval.compute_traces(shape[0], shape[1], g2)
    return np.abs(max_pool(t1, sigma) - max_pool(t2, sigma))

def genome_compatibility(g1: Genome, g2: Genome) -> float:
    if g1.population is None:
        raise ValueError("Genome has no population reference.")
    return calculate_compatibility(g1.population, g1, g2)

def get_best_from_species(pop: Population, n_species: Optional[int] = None) -> List[Genome]:
    get_best = lambda gs: max(gs, key=lambda g: g.fitness)
    genomes = sorted([get_best(s.members) for s in pop.species],
                     key=lambda g: g.fitness, reverse=True)
    return genomes if n_species is None else genomes[:n_species]


def genome_to_network(genome: Genome) -> Network:
    nodes = [
        NetworkNode(id=nid, pos=_get_node_pos(n), type=_get_node_type(n))
        for nid, n in genome.nodes.items()
    ]
    edges = [
        NetworkEdge(id=e.id, node1=e.node1, node2=e.node2, enabled=e.enabled)
        for e in genome.edges.values()
    ]
    return Network(nodes=nodes, edges=edges, fitness=genome.fitness)


def network_to_genome(network: Network, pop: Population) -> Genome:
    node_dict = {
        n.id: Node(id=n.id, data=NodeData(type_=n.type, pos=n.pos))
        for n in network.nodes
    }
    edge_dict = {
        e.id: Edge(id=e.id, node1=e.node1, node2=e.node2, enabled=e.enabled)
        for e in network.edges
    }
    return Genome(
        config=pop.config,
        specialization=pop.specialization,
        population=pop,
        nodes=node_dict,
        edges=edge_dict,
    )


@dataclass
class RunResults:
    top_networks: List[Network]
    pop: Population
    stats: RunStatistics
    best_from_species: List[Network]


def run(
    config: NEATCoreConfig,
    eval_config: EvalConfig,
    print_: bool = True,
) -> RunResults:
    from neatwork.neatwork_py_eval import evaluate_batch

    def f_eval(genomes):
        return evaluate_batch(
            eval_config.cost_map, genomes,
            objectives=[o.value for o in eval_config.objectives] or None,
            scalarization=eval_config.scalarization,
            constraints=[c.value for c in eval_config.constraints],
        )

    if config.start_genome is not None:
        temp_pop = neat_core_py.Population(config, NEATWork)
        start_genome = network_to_genome(config.start_genome, temp_pop)
        config = config.model_copy(update={"start_genome": start_genome})

    top_genomes, pop, stats = neat_core_py.run(
        config=config,
        f_eval=f_eval,
        specialization_cls=NEATWork,
        print_=print_,
    )

    return RunResults(
        top_networks=[genome_to_network(g) for g in top_genomes],
        pop=pop,
        stats=stats,
        best_from_species=[genome_to_network(g) for g in get_best_from_species(pop)],
    )
