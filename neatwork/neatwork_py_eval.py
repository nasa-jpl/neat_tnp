import heapq
from typing import List, Dict, Tuple
import numpy as np
from skimage.draw import line_nd

from neatwork.neat_core_py import *
import neatwork.neatwork_py as neatwork_py

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
    return neatwork_py._get_node_pos(node)

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

def connected(genome: Genome) -> bool:
    return neatwork_py.check_connectivity(
        nodes=list(genome.nodes.keys()),
        edges={
            edge.id: (edge.node1, edge.node2)
            for edge in genome.edges.values()
            if edge.enabled
        }
    )

def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def _edges_intersect(A, B, C, D):
    # Ignore shared endpoints
    if A == C or A == D or B == C or B == D:
        return False
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)

def has_intersecting_edges(genome: Genome) -> bool:
    edges = []
    for e in genome.edges.values():
        if e.enabled:
            p1 = _get_node_pos(genome.nodes[e.node1])
            p2 = _get_node_pos(genome.nodes[e.node2])
            edges.append((p1, p2))
    
    n = len(edges)
    for i in range(n):
        for j in range(i + 1, n):
            if _edges_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return True
    return False

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
    return -loss

def calculate_traffic_balance(genome: Genome) -> float:
    nodes = list(genome.nodes.values())
    if len(nodes) < 2: return 0.0
    
    id_map = {n.id: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = [[] for _ in range(n)]
    
    is_fixed = [neatwork_py._get_node_type(node) == neatwork_py.NodeType.Fixed for node in nodes]
    total_terminals = sum(1 for f in is_fixed if f)
    
    for e in genome.edges.values():
        if e.enabled:
            u = id_map[e.node1]
            v = id_map[e.node2]
            adj[u].append(v)
            adj[v].append(u)
            
    tin = [-1] * n
    low = [-1] * n
    timer = 0
    total_severity = [0.0]
    
    def dfs(u, p):
        nonlocal timer
        tin[u] = low[u] = timer
        timer += 1
        
        st = 1 if is_fixed[u] else 0
        
        for v in adj[u]:
            if v == p: continue
            
            if tin[v] != -1:
                low[u] = min(low[u], tin[v])
            else:
                st_v = dfs(v, u)
                st += st_v
                low[u] = min(low[u], low[v])
                
                if low[v] > tin[u]:
                    # Bridge!
                    out_term = total_terminals - st_v
                    severity = abs(st_v - out_term)
                    total_severity[0] += severity
        return st
        
    for i in range(n):
        if tin[i] == -1:
            dfs(i, -1)
            
    return -float(total_severity[0])

def get_edge_cost(p1, p2, cost_map: np.ndarray) -> float:
    if np.allclose(p1, p2): return 0.0
    rows, cols = line_nd(p1, p2, endpoint=True)
    H, W = cost_map.shape
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows, cols = rows[valid], cols[valid]
    if len(rows) < 2: return 0.0
    
    dp = np.array(p1) - np.array(p2)
    edge_vec = np.abs(dp)
    edge_vec = edge_vec / np.linalg.norm(dp)
    di, dj = edge_vec
    a, b = max(di, dj), min(di, dj)
    c_div_a = np.sqrt(a**2 + b**2)/a if a > 0 else 1.0
    
    vals = cost_map[rows, cols]
    return float(np.sum(vals) * c_div_a)

def calculate_max_edge_traffic(genome: Genome, cost_map: np.ndarray) -> float:
    nodes = list(genome.nodes.values())
    if len(nodes) < 2: return 0.0
    
    id_map = {n.id: i for i, n in enumerate(nodes)}
    n = len(nodes)
    terminals = [i for i, n in enumerate(nodes) if neatwork_py._get_node_type(n) == neatwork_py.NodeType.Fixed]
    if len(terminals) < 2: return 0.0
    
    adj = [[] for _ in range(n)]
    edge_marks = {}
    
    for e in genome.edges.values():
        if not e.enabled: continue
        p1 = neatwork_py._get_node_pos(genome.nodes[e.node1])
        p2 = neatwork_py._get_node_pos(genome.nodes[e.node2])
        
        cost = get_edge_cost(p1, p2, cost_map)
        
        u = id_map[e.node1]
        v = id_map[e.node2]
        adj[u].append((v, e.id, cost))
        adj[v].append((u, e.id, cost))
        edge_marks[e.id] = 0
        
    for i in range(len(terminals)):
        start = terminals[i]
        dists = [float('inf')] * n
        preds = [None] * n
        dists[start] = 0.0
        
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dists[u]: continue
            
            for v, eid, ecost in adj[u]:
                nxt_cost = d + ecost
                if nxt_cost < dists[v]:
                    dists[v] = nxt_cost
                    preds[v] = (u, eid)
                    heapq.heappush(pq, (nxt_cost, v))
                    
        for j in range(i + 1, len(terminals)):
            curr = terminals[j]
            while curr != start:
                pred = preds[curr]
                if pred is not None:
                    prev, eid = pred
                    edge_marks[eid] += 1
                    curr = prev
                else:
                    break
                    
    max_marks = max(edge_marks.values()) if edge_marks else 0
    return -float(max_marks)

def _compute_objective(name: str, cost_map, genome: Genome) -> float:
    """
    Compute a single named objective for a genome.

    TODO: implement each objective branch.
    """
    if name == "cost_map" or name == neatwork_py.Objective.CostMap:
        return loss_to_fit(weight_on_cost_map(cost_map, genome))
    elif name == "traffic_balance" or name == neatwork_py.Objective.TrafficBalance:
        return calculate_traffic_balance(genome)
    elif name == "max_edge_traffic" or name == neatwork_py.Objective.MaxEdgeTraffic:
        return calculate_max_edge_traffic(genome, cost_map)
    raise NotImplementedError(f"Objective {name!r} is not implemented yet")

def _scalarize(values: List[float], strategy) -> float:
    """
    Combine a list of objective values into a single scalar fitness.
    """
    if isinstance(strategy, dict) and "augmented_chebyshev" in strategy:
        rho = strategy["augmented_chebyshev"].get("rho", 0.05)
        # Note: here we simulate the scalarizer from python reference but we do not have
        # utopia/nadir tracked across population. The real evaluate_batch does that!
        raise ValueError("Scalarize cannot be called globally without population bounds in python reference")
        
    raise NotImplementedError(f"ScalarizationStrategy {strategy!r} is not implemented yet")


def evaluate(
    cost_map,
    genome: Genome,
    corrected_matrix_trajectories: bool = True,
    objectives: List[str] | None = None,
    scalarization: str | None = None,
) -> float:
    # Deprecated for MOO since scalarization must occur over population
    raise DeprecationWarning("Please use evaluate_batch for correct scalarization handling")


def evaluate_batch(
    cost_map,
    genomes: List[Genome],
    corrected_matrix_trajectories: bool = True,
    objectives: List[str] | None = None,
    scalarization = None,
    constraints: List[str] | None = None,
) -> List[float]:
    if constraints is None:
        constraints = []

    def fails_optional_constraints(genome) -> bool:
        for c in constraints:
            if c == "no_intersecting_edges" and has_intersecting_edges(genome):
                return True
        return False

    if not objectives:
        # Default single objective
        for genome in genomes:
            if not connected(genome) or fails_optional_constraints(genome):
                genome.fitness = -float('inf')
            else:
                if corrected_matrix_trajectories:
                    genome.fitness = loss_to_fit(weight_on_cost_map(cost_map, genome))
                else:
                    genome.fitness = loss_to_fit(weight_on_cost_map_old(cost_map, genome))
        return [g.fitness for g in genomes]

    n_objs = len(objectives)

    # Phase 1: Compute raw
    valid_mask = []
    raw_results = []
    for g in genomes:
        if not connected(g) or fails_optional_constraints(g):
            valid_mask.append(False)
            raw_results.append(None)
            g.fitness = -float('inf')
        else:
            valid_mask.append(True)
            vals = [_compute_objective(obj, cost_map, g) for obj in objectives]
            raw_results.append(vals)
            
    # Phase 2: Scalarize
    if n_objs == 1:
        for i, g in enumerate(genomes):
            if valid_mask[i]:
                g.fitness = raw_results[i][0]
                
    else:
        # compute bounds
        utopia = [-float('inf')] * n_objs
        nadir = [float('inf')] * n_objs
        
        for res in raw_results:
            if res is not None:
                for j in range(n_objs):
                    utopia[j] = max(utopia[j], res[j])
                    nadir[j] = min(nadir[j], res[j])
                    
        rho = 0.05
        if isinstance(scalarization, dict) and "augmented_chebyshev" in scalarization:
            rho = scalarization["augmented_chebyshev"].get("rho", 0.05)
            
        for i, g in enumerate(genomes):
            if valid_mask[i]:
                vals = raw_results[i]
                max_weighted_dist = -float('inf')
                sum_weighted_dist = 0.0
                
                for j in range(n_objs):
                    domain_range = utopia[j] - nadir[j]
                    w = 1.0 / domain_range if domain_range > 1e-9 else 1.0
                    
                    dist = utopia[j] - vals[j]
                    weighted_dist = w * dist
                    
                    if weighted_dist > max_weighted_dist:
                        max_weighted_dist = weighted_dist
                    sum_weighted_dist += weighted_dist
                    
                scalar_dist = max_weighted_dist + (rho * sum_weighted_dist)
                g.fitness = -scalar_dist
                
    return [g.fitness for g in genomes]
