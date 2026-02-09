from typing import List, Dict, Tuple
import numpy as np
from skimage.draw import line_nd

from neat_tnp.neat_py_core import *
import neat_tnp.neat_py_tnp as neat_py_tnp

def matrix_trajectories_basic(mat: np.ndarray, p1: np.ndarray, p2: np.ndarray, avg: bool = False) -> float:
    """
    An old (and slightly flawed) way of computing cost map coverage.
    This is retained as an example showing now NEAT learns to cheat.
    """
    if np.allclose(p1, p2):
        return 0.0
    rows, cols = line_nd(p1, p2)
    # filter out-of-bounds indices
    H, W = mat.shape
    valid_mask = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows, cols = rows[valid_mask], cols[valid_mask]
    # sample matrix values
    line_cell_values = mat[rows, cols]
    assert not np.isnan(np.mean(line_cell_values)), f"cannot compute trace.\np1,p2: {p1,p2}\nmat: {mat}\nrows,cols: {rows,cols}"
    if avg:
        return float(np.mean(np.exp(line_cell_values)))
        # return np.mean(line_cell_values)
    return float(np.sum(line_cell_values))

def _get_node_pos(node: Node) -> Tuple[int, int]:
    return neat_py_tnp._get_node_pos(node)

def compute_traces(h: int, w: int, genome: Genome) -> np.ndarray:
    traces = np.zeros((h, w), dtype=np.float64)
    endpoints = np.zeros((h, w), dtype=np.float64)

    for node in genome.nodes.values():
        r, c = _get_node_pos(node)
        if 0 <= r < h and 0 <= c < w:
            traces[r, c] += 1.0

    for edge in genome.edges.values():
        if not edge.enabled:
            continue
        n1 = genome.nodes[edge.node1]
        n2 = genome.nodes[edge.node2]
        r0, c0 = _get_node_pos(n1)
        r1, c1 = _get_node_pos(n2)

        dist = np.sqrt((r0 - r1) ** 2 + (c0 - c1) ** 2)
        if dist < 1e-5:
            continue

        n_steps = int(np.ceil(max(abs(r0 - r1), abs(c0 - c1)))) + 1
        if n_steps < 2:
            continue

        edge_vec = (abs(r1 - r0), abs(c1 - c0))
        a = max(edge_vec)
        b = min(edge_vec)
        c_div_a = np.sqrt(a ** 2 + b ** 2) / a if a > 0 else 1.0
        weight = c_div_a

        for i in range(n_steps):
            t = i / (n_steps - 1)
            r = int(round(r0 * (1.0 - t) + r1 * t))
            c = int(round(c0 * (1.0 - t) + c1 * t))
            if 0 <= r < h and 0 <= c < w:
                if i == 0 or i == n_steps - 1:
                    endpoints[r, c] += weight
                else:
                    traces[r, c] += weight

    node_mask = endpoints > 0.0
    traces[node_mask] += endpoints[node_mask]
    traces /= (h + w)

    return traces

def matrix_trajectories_mask(mat: np.ndarray, edges: List[Tuple[Tuple[int,int],Tuple[int,int]]]):
    """
    Creates a mask of coverage of the provided edges over a matrix of shape mat.

    All contributions are scaled by the size of mat.
    """
    
    H, W = mat.shape

    # the coverage of all the edges in mat
    traces = np.zeros_like(mat)
    n_endpoints = np.zeros_like(mat)
    incident_edge_weights = np.zeros_like(mat)
    
    for p1,p2 in edges:
        if np.allclose(p1, p2):
            continue
        
        rows, cols = line_nd(p1, p2, endpoint=True)
        
        # filter out-of-bounds indices
        valid_mask = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows, cols = rows[valid_mask], cols[valid_mask]
        n_points = len(rows)
        if n_points < 2:
            continue

        # remove the endpoints from the trace
        p1_row, p1_col = p1
        p2_row, p2_col = p2
        is_p1_mask = (rows == p1_row) & (cols == p1_col)
        is_p2_mask = (rows == p2_row) & (cols == p2_col)
        is_endpoint_mask = is_p1_mask | is_p2_mask
        edge_rows, edge_cols = rows[~is_endpoint_mask], cols[~is_endpoint_mask]
        end_rows, end_cols = rows[is_endpoint_mask], cols[is_endpoint_mask]
        
        weight = 1

        # ensure the traces respect euclidean distances
        #   * horizontal/vertical line  ->  c=1
        #   * diagonal line:            ->  c=sqrt(2)
        edge_vec = np.abs(np.array(p1) - np.array(p2))
        edge_vec = edge_vec / np.linalg.norm(edge_vec)
        di, dj = tuple(edge_vec)
        a, b = max(di, dj), min(di, dj)
        c_div_a = np.sqrt(a**2 + b**2)/a
        weight *= c_div_a

        # store traces
        traces[edge_rows,edge_cols] += weight
        n_endpoints[end_rows, end_cols] += 1
        incident_edge_weights[end_rows, end_cols] += weight
    
    # add contributions of edpoints (nodes)
    node_mask = n_endpoints>0
    traces[node_mask] += incident_edge_weights[node_mask]#/n_endpoints[node_mask]

    # scale the mask by the size of the map
    # TODO: is it more numerically stable to apply this 
    #       scaling after computing the sum of losses?
    traces /= H+W

    # print(traces)

    return traces

def matrix_trajectories_masked(mat: np.ndarray, edges: List[Tuple[Tuple[int,int],Tuple[int,int]]]) -> float:
    total_coverage: np.ndarray = matrix_trajectories_mask(mat, edges)
    cost_map_masked = np.multiply(mat, total_coverage)
    final_sum = np.sum(cost_map_masked)
    assert not np.isnan(final_sum), f"cannot compute traces"
    return float(final_sum)

def bridge_share(nodes: Dict[NodeID, Node], edges: Dict[EdgeID, Edge]) -> float:
    node_ids = list(nodes.keys())
    if len(node_ids) < 2:
        return 0.0

    id_map = {node_id: i for i, node_id in enumerate(node_ids)}
    coords = [_get_node_pos(nodes[node_id]) for node_id in node_ids]

    adj: List[List[Tuple[int, Edge]]] = [[] for _ in node_ids]
    total_len = 0.0
    for edge in edges.values():
        if not edge.enabled:
            continue
        u = id_map[edge.node1]
        v = id_map[edge.node2]
        adj[u].append((v, edge))
        adj[v].append((u, edge))
        p1 = coords[u]
        p2 = coords[v]
        total_len += np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    if total_len < 1e-9:
        return 0.0

    n = len(node_ids)
    tin = [-1] * n
    low = [-1] * n
    timer = 0
    bridge_len = 0.0

    # Iterative Tarjan bridge-finding
    # Stack frames: (node, parent, neighbor_index)
    stack = [(0, -1, 0)]
    tin[0] = timer
    low[0] = timer
    timer += 1

    while stack:
        u, p, ni = stack[-1]
        if ni < len(adj[u]):
            stack[-1] = (u, p, ni + 1)
            v, _ = adj[u][ni]
            if v == p:
                continue
            if tin[v] != -1:
                low[u] = min(low[u], tin[v])
            else:
                tin[v] = timer
                low[v] = timer
                timer += 1
                stack.append((v, u, 0))
        else:
            stack.pop()
            if stack:
                parent_u = stack[-1][0]
                low[parent_u] = min(low[parent_u], low[u])
                if low[u] > tin[parent_u]:
                    p1 = coords[parent_u]
                    p2 = coords[u]
                    bridge_len += np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    return bridge_len / total_len

def connected(genome: Genome) -> bool:
    return neat_py_tnp.check_connectivity(
        nodes=list(genome.nodes.keys()),
        edges={
            edge.id: (edge.node1, edge.node2)
            for edge in genome.edges.values()
            if edge.enabled
        }
    )

def weight_on_cost_map(cost_map, genome: Genome) -> float:
    traces = compute_traces(cost_map.shape[0], cost_map.shape[1], genome)
    cost_map_masked = np.multiply(cost_map, traces)
    final_sum = np.sum(cost_map_masked)
    assert not np.isnan(final_sum), f"cannot compute traces"
    return float(final_sum)

def weight_on_cost_map_old(cost_map, genome: Genome) -> float:
    loss = 0.0
    for edge in genome.edges.values():
        if not edge.enabled:
            continue
        p1 = _get_node_pos(genome.nodes[edge.node1])
        p2 = _get_node_pos(genome.nodes[edge.node2])
        weight = matrix_trajectories_basic(
            mat=cost_map, 
            p1=np.array(p1), 
            p2=np.array(p2), 
            avg=False,
        )
        loss += weight
    return loss

def loss_to_fit(loss: float) -> float:
    eps = 1e-7
    loss = max(loss, eps)
    fitness = 1/loss
    return fitness

def evaluate(cost_map, genome: Genome, corrected_matrix_trajectories=True) -> float:
    if not connected(genome):
        fitness = 0.0
    else:
        if corrected_matrix_trajectories:
            fitness = loss_to_fit(weight_on_cost_map(cost_map, genome))
        else:
            fitness = loss_to_fit(weight_on_cost_map_old(cost_map, genome))
    
    genome.fitness = fitness
    return genome.fitness

def evaluate_multi(cost_map, genome: Genome) -> Tuple[float, ...]:
    if not connected(genome):
        objectives = (0.0, 0.0)
    else:
        cost_map_fit = loss_to_fit(weight_on_cost_map(cost_map, genome))
        robustness = 1.0 - bridge_share(genome.nodes, genome.edges)
        objectives = (cost_map_fit, robustness)

    genome.fitness = objectives[0]
    genome.objectives = objectives
    return objectives

def evaluate_batch(cost_map, genomes: List[Genome], corrected_matrix_trajectories=True) -> List[float]:
    return [evaluate(cost_map, genome, corrected_matrix_trajectories) for genome in genomes]

def evaluate_multi_batch(cost_map, genomes: List[Genome]) -> List[Tuple[float]]:
    return [evaluate_multi(cost_map, genome) for genome in genomes]
