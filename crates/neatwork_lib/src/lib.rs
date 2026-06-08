use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyReadonlyArray2, ToPyArray};
use ndarray::Array2;
use neat_core_rs::{
    Genome as CoreGenome, 
    Population as CorePopulation,
    Specialization, 
    Node as CoreNode, 
    Edge as CoreEdge,
    NEATConfig as CoreNEATConfig,
};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use pythonize::{depythonize, pythonize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};


// ============================================================================
// 1. DOMAIN TYPES (Unified Rust + Python + Serde)
// ============================================================================

mod types {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub enum NodeType { Fixed, Flexible }
    impl Default for NodeType { fn default() -> Self { NodeType::Flexible } }

    #[derive(Clone, Debug, Default, Copy, Serialize, Deserialize)]
    pub struct NWNodeData {
        pub pos: (i32, i32),
        #[serde(rename = "type")]
        pub type_: NodeType,
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    pub struct NWEdgeData;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub enum GraphInit { FCN, Grid(u32, u32) }
    impl Default for GraphInit { fn default() -> Self { Self::FCN } }

    /// Strategy for combining multiple objective values into a single fitness.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub enum ScalarizationStrategy {
        #[serde(rename = "augmented_chebyshev")]
        AugmentedChebyshev {
            rho: f64,
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct NWConfig {
        pub grid_size: (usize, usize),
        pub fixed_nodes: Vec<(i32, i32)>,
        pub move_node_mutation_prob: f64,
        pub move_node_mutation_sigma: f64,
        pub graph_init: GraphInit,
        pub compatibility_sigma: usize,
        pub c3: f64,
    }

    #[derive(Clone, Debug)]
    pub struct NWSpecialization { pub config: NWConfig }
}

use types::*;

// ============================================================================
// 2. ALGORITHM (Grid & Specialization)
// ============================================================================

mod algorithm {
    use super::*;

    pub struct Grid { h: i32, w: i32 }

    impl Grid {
        pub fn new(size: (usize, usize)) -> Self { 
            Self { h: size.0 as i32, w: size.1 as i32 } 
        }
        
        pub fn clip(&self, pos: (i32, i32)) -> (i32, i32) {
            (pos.0.max(0).min(self.h - 1), pos.1.max(0).min(self.w - 1))
        }

        pub fn is_occupied(
            &self, 
            pos: (i32, i32), 
            nodes: &HashMap<u64, CoreNode<NWNodeData>>, 
            ignore: Option<u64>
        ) -> bool {
            nodes.values().any(|n| n.data.pos == pos && Some(n.id) != ignore)
        }

        pub fn find_free_pos(
            &self, 
            center: (i32, i32), 
            nodes: &HashMap<u64, CoreNode<NWNodeData>>, 
            rng: &mut impl Rng
        ) -> Option<(i32, i32)> {
            if !self.is_occupied(center, nodes, None) { return Some(center); }
            
            let max_rad = self.h.max(self.w);
            for r in 1..max_rad {
                for _ in 0..10 {
                    let dr = rng.gen_range(-r..=r);
                    let dc = rng.gen_range(-r..=r);
                    let pos = self.clip((center.0 + dr, center.1 + dc));
                    if !self.is_occupied(pos, nodes, None) { return Some(pos); }
                }
            }
            None
        }
    }

    impl Specialization for NWSpecialization {
        type NodeData = NWNodeData;
        type EdgeData = NWEdgeData;
        type Config = NWConfig;

        fn normalize_edge(&self, n1: u64, n2: u64) -> (u64, u64) { 
            if n1 < n2 { (n1, n2) } else { (n2, n1) } 
        }

        fn initialize_genome(
            &self, 
            pop: &CorePopulation<Self>
        ) -> (Vec<CoreNode<Self::NodeData>>, Vec<CoreEdge<Self::EdgeData>>) {
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let state = &pop.state;
            let grid = Grid::new(self.config.grid_size);

            // 1. Fixed Nodes (Always present)
            for &pos in &self.config.fixed_nodes {
                nodes.push(CoreNode::new(
                    state.new_node_id(), 
                    Some(NWNodeData { pos, type_: NodeType::Fixed })
                ));
            }

            match self.config.graph_init {
                GraphInit::FCN => {
                    // Fully connect fixed nodes
                    for i in 0..nodes.len() {
                        for j in (i + 1)..nodes.len() {
                            let (n1, n2) = self.normalize_edge(nodes[i].id, nodes[j].id);
                            edges.push(CoreEdge::new(
                                n1, n2, 
                                state.get_edge_id(self, n1, n2), 
                                Some(true), 
                                None
                            ));
                        }
                    }
                }
                GraphInit::Grid(rows, cols) => {
                    let mut grid_ids = HashMap::new();
                    let mut flex_pos = Vec::new();
                    let mut node_map = HashMap::new();
                    let mut rng = rand::thread_rng();

                    // 1. Flexible grid nodes
                    let (step_r, step_c) = (
                        grid.h as f64 / rows as f64, 
                        grid.w as f64 / cols as f64
                    );
                    for r in 0..rows {
                        for c in 0..cols {
                            let ideal = (
                                ((r as f64 + 0.5) * step_r).round() as i32,
                                ((c as f64 + 0.5) * step_c).round() as i32
                            );
                            
                            let pos = if grid.is_occupied(ideal, &node_map, None) {
                                if let Some(p) = grid.find_free_pos(ideal, &node_map, &mut rng) { 
                                    p 
                                } else { 
                                    continue; 
                                }
                            } else {
                                ideal
                            };
                            
                            let id = state.new_node_id();
                            let node = CoreNode::new(
                                id, 
                                Some(NWNodeData { pos, type_: NodeType::Flexible })
                            );
                            nodes.push(node.clone());
                            node_map.insert(id, node);
                            grid_ids.insert((r, c), id);
                            flex_pos.push((id, pos));
                        }
                    }

                    // 2. Grid edges
                    for r in 0..rows {
                        for c in 0..cols {
                            if let Some(&u) = grid_ids.get(&(r, c)) {
                                for (dr, dc) in &[(0, 1), (1, 0)] {
                                    if let Some(&v) = grid_ids.get(&(r + dr, c + dc)) {
                                        let (n1, n2) = self.normalize_edge(u, v);
                                        edges.push(CoreEdge::new(
                                            n1, n2, 
                                            state.get_edge_id(self, n1, n2), 
                                            Some(true), 
                                            None
                                        ));
                                    }
                                }
                            }
                        }
                    }

                    // 3. Connect fixed nodes to nearest flexible
                    for fixed_node in nodes.iter().filter(|n| n.data.type_ == NodeType::Fixed) {
                        if let Some((best_id, _)) = flex_pos.iter().min_by_key(|(_, pos)| {
                            // compute distance
                            let fpos = fixed_node.data.pos;
                            (fpos.0 - pos.0).pow(2) + (fpos.1 - pos.1).pow(2)
                        }) {
                            let (n1, n2) = self.normalize_edge(fixed_node.id, *best_id);
                            edges.push(CoreEdge::new(
                                n1, n2, 
                                state.get_edge_id(self, n1, n2), 
                                Some(true), 
                                None
                            ));
                        }
                    }
                }
            }
            (nodes, edges)
        }

        fn mutate_structure(
            &self, 
            nodes: &mut HashMap<u64, CoreNode<Self::NodeData>>, 
            _: &mut HashMap<u64, CoreEdge<Self::EdgeData>>, 
            rng: &mut impl Rng
        ) {
            if !rng.gen_bool(self.config.move_node_mutation_prob) { return; }
            
            let flex_nodes: Vec<u64> = nodes.iter()
                .filter(|(_, n)| n.data.type_ == NodeType::Flexible)
                .map(|(k, _)| *k)
                .collect();
                
            if let Some(&id) = flex_nodes.choose(rng) {
                let old_pos = nodes[&id].data.pos;
                let s = self.config.move_node_mutation_sigma;
                
                let dr = (rng.sample::<f64, _>(StandardNormal) * s).round() as i32;
                let dc = (rng.sample::<f64, _>(StandardNormal) * s).round() as i32;
                
                let grid = Grid::new(self.config.grid_size);
                let new_pos = grid.clip((old_pos.0 + dr, old_pos.1 + dc));
                
                if !grid.is_occupied(new_pos, nodes, Some(id)) {
                    if let Some(n) = nodes.get_mut(&id) { 
                        n.data.pos = new_pos;
                    }
                }
            }
        }

        fn add_node(
            &self, 
            new_node: &mut CoreNode<Self::NodeData>, 
            old_edge: &CoreEdge<Self::EdgeData>, 
            _: &mut CoreEdge<Self::EdgeData>, 
            _: &mut CoreEdge<Self::EdgeData>, 
            nodes: &HashMap<u64, CoreNode<Self::NodeData>>, 
            _: &HashMap<u64, CoreEdge<Self::EdgeData>>, 
            rng: &mut impl Rng
        ) -> bool {
            let p1 = nodes[&old_edge.node1].data.pos;
            let p2 = nodes[&old_edge.node2].data.pos;
            let grid = Grid::new(self.config.grid_size);
            
            let midpoint = ((p1.0 + p2.0) / 2, (p1.1 + p2.1) / 2);
            if let Some(pos) = grid.find_free_pos(midpoint, nodes, rng) {
                new_node.data = NWNodeData { pos, type_: NodeType::Flexible };
                true
            } else { 
                false 
            }
        }

        fn compatibility(
            &self, 
            g1: &CoreGenome<Self>, 
            g2: &CoreGenome<Self>, 
            _: &[u64]
        ) -> f64 {
            let (h, w) = self.config.grid_size;
            let factor = self.config.c3;
            // No 'mut' needed here anymore, we read-only into pool
            let t1 = evaluation::compute_traces(h, w, g1);
            let t2 = evaluation::compute_traces(h, w, g2);

            let s1 = evaluation::max_pool(&t1, self.config.compatibility_sigma);
            let s2 = evaluation::max_pool(&t2, self.config.compatibility_sigma);
            
            s1.iter().zip(s2.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
            * factor
        }

        fn add_edge(
            &self, 
            _: &mut CoreEdge<Self::EdgeData>, 
            _: &HashMap<u64, CoreNode<Self::NodeData>>, 
            _: &HashMap<u64, CoreEdge<Self::EdgeData>>, 
            _: &mut impl Rng
        ) -> bool { true }
        
        fn remove_node(
            &self, 
            n: &CoreNode<Self::NodeData>, 
            _: &CoreEdge<Self::EdgeData>, 
            _: &CoreEdge<Self::EdgeData>, 
            _new_edge: &CoreEdge<Self::EdgeData>, 
            nodes: &HashMap<u64, CoreNode<Self::NodeData>>, 
            edges: &HashMap<u64, CoreEdge<Self::EdgeData>>, 
            _: &mut impl Rng
        ) -> bool { 
            if n.data.type_ != NodeType::Flexible { return false; }
            
            let remaining_nodes: Vec<u64> = nodes.keys()
                .copied()
                .filter(|&nid| nid != n.id)
                .collect();
            let remaining_edges: HashMap<u64, (u64, u64)> = edges.iter()
                .filter(|(_, e)| e.enabled && e.node1 != n.id && e.node2 != n.id)
                .map(|(&k, e)| (k, (e.node1, e.node2)))
                .collect();
            
            evaluation::check_connectivity_dict(&remaining_nodes, &remaining_edges)
        }
        
        fn remove_node_single(
            &self, 
            n: &CoreNode<Self::NodeData>, 
            _: &CoreEdge<Self::EdgeData>, 
            nodes: &HashMap<u64, CoreNode<Self::NodeData>>, 
            edges: &HashMap<u64, CoreEdge<Self::EdgeData>>, 
            _: &mut impl Rng
        ) -> bool { 
            if n.data.type_ != NodeType::Flexible { return false; }
            
            let remaining_nodes: Vec<u64> = nodes.keys()
                .copied()
                .filter(|&nid| nid != n.id)
                .collect();
            let remaining_edges: HashMap<u64, (u64, u64)> = edges.iter()
                .filter(|(_, e)| e.enabled && e.node1 != n.id && e.node2 != n.id)
                .map(|(&k, e)| (k, (e.node1, e.node2)))
                .collect();
            
            evaluation::check_connectivity_dict(&remaining_nodes, &remaining_edges)
        }
        
        fn remove_edge(
            &self, 
            edge: &CoreEdge<Self::EdgeData>, 
            nodes: &HashMap<u64, CoreNode<Self::NodeData>>, 
            edges: &HashMap<u64, CoreEdge<Self::EdgeData>>, 
            _: &mut impl Rng
        ) -> bool {
            let all_nodes: Vec<u64> = nodes.keys().copied().collect();
            let remaining_edges: HashMap<u64, (u64, u64)> = edges.iter()
                .filter(|(&eid, e)| e.enabled && eid != edge.id)
                .map(|(&k, e)| (k, (e.node1, e.node2)))
                .collect();
            
            evaluation::check_connectivity_dict(&all_nodes, &remaining_edges)
        }
    }
}

// ============================================================================
// 3. EVALUATION (Math & Traces)
// ============================================================================

mod evaluation {
    use super::*;

    pub fn check_connectivity_dict(
        nodes: &[u64], 
        edges: &HashMap<u64, (u64, u64)>
    ) -> bool {
        if nodes.is_empty() { return true; }
        
        let mut adj = HashMap::new();
        for &(n1, n2) in edges.values() {
            adj.entry(n1).or_insert_with(Vec::new).push(n2);
            adj.entry(n2).or_insert_with(Vec::new).push(n1);
        }
        
        let start = nodes[0];
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        visited.insert(start);
        
        while let Some(u) = stack.pop() {
            if let Some(nb) = adj.get(&u) {
                for &v in nb { 
                    if visited.insert(v) { stack.push(v); } 
                }
            }
        }
        visited.len() == nodes.len()
    }

    fn ccw(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
        (c.1 - a.1) * (b.0 - a.0) > (b.1 - a.1) * (c.0 - a.0)
    }

    fn edges_intersect(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
        if a == c || a == d || b == c || b == d {
            return false;
        }
        ccw(a, c, d) != ccw(b, c, d) && ccw(a, b, c) != ccw(a, b, d)
    }

    pub fn has_intersecting_edges(g: &CoreGenome<NWSpecialization>) -> bool {
        let mut segments = Vec::new();
        for e in g.edges.values() {
            if e.enabled {
                if let (Some(n1), Some(n2)) = (g.nodes.get(&e.node1), g.nodes.get(&e.node2)) {
                    let p1 = (n1.data.pos.0 as f64, n1.data.pos.1 as f64);
                    let p2 = (n2.data.pos.0 as f64, n2.data.pos.1 as f64);
                    segments.push((p1, p2));
                }
            }
        }
        let n = segments.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if edges_intersect(segments[i].0, segments[i].1, segments[j].0, segments[j].1) {
                    return true;
                }
            }
        }
        false
    }


    /// Note: This is technically a floating-point Digital Differential Analyzer (DDA)
    /// algorithm, rather than pure integer Bresenham, to ensure perfectly symmetric
    /// edge-weights and reproduce the original Python implementation's trace properties.
    #[inline]
    pub fn bresenham(
        pos1: (i32, i32), 
        pos2: (i32, i32), 
        mut callback: impl FnMut(usize, usize, f64, bool)
    ) {
        let (r0, c0) = (pos1.0 as f64, pos1.1 as f64);
        let (r1, c1) = (pos2.0 as f64, pos2.1 as f64);
        
        let dist = ((r0 - r1).powi(2) + (c0 - c1).powi(2)).sqrt();
        if dist < 1e-5 { return; }
        
        let n_steps = (r0 - r1).abs().max((c0 - c1).abs()).ceil() as usize + 1;
        if n_steps < 2 { return; }
        
        // Weight respects euclidean distance
        let edge_vec = ((r1 - r0).abs(), (c1 - c0).abs());
        let (a, b) = (edge_vec.0.max(edge_vec.1), edge_vec.0.min(edge_vec.1));
        let c_div_a = if a > 0.0 { (a.powi(2) + b.powi(2)).sqrt() / a } else { 1.0 };
        let weight = c_div_a;
        
        for i in 0..n_steps {
            let t = i as f64 / (n_steps as f64 - 1.0);
            let r = (r0 * (1.0 - t) + r1 * t).round() as usize;
            let c = (c0 * (1.0 - t) + c1 * t).round() as usize;
            
            let is_endpoint = i == 0 || i == n_steps - 1;
            callback(r, c, weight, is_endpoint);
        }
    }

    pub fn compute_traces(
        h: usize, 
        w: usize, 
        g: &CoreGenome<NWSpecialization>
    ) -> Array2<f64> {
        let mut traces = Array2::<f64>::zeros((h, w));
        let mut endpoints = Array2::<f64>::zeros((h, w));
        
        // 1. Node contributions
        for n in g.nodes.values() {
            let (r, c) = (n.data.pos.0 as usize, n.data.pos.1 as usize);
            if r < h && c < w { traces[[r, c]] += 1.0; }
        }

        // 2. Edge contributions
        for e in g.edges.values() {
            if !e.enabled { continue; }
            let n1 = &g.nodes[&e.node1];
            let n2 = &g.nodes[&e.node2];
            
            bresenham(n1.data.pos, n2.data.pos, |r, c, weight, is_endpoint| {
                if r < h && c < w {
                    if is_endpoint { 
                        endpoints[[r, c]] += weight; 
                    } else { 
                        traces[[r, c]] += weight; 
                    }
                }
            });
        }
        
        // 3. Merge endpoints
        for ((r, c), &v) in endpoints.indexed_iter() { 
            if v > 0.0 { traces[[r, c]] += v; } 
        }
        
        // 4. Scale by map size
        let scale = 1.0 / ((h + w) as f64);
        traces.mapv_inplace(|v| v * scale);
        traces
    }

    pub fn max_pool(input: &Array2<f64>, block: usize) -> Array2<f64> {
        let (h, w) = input.dim();
        let (nh, nw) = (h / block, w / block);
        let mut out = Array2::zeros((nh, nw));
        for r in 0..nh {
            for c in 0..nw {
                let mut max_v = 0.0f64;
                for ir in 0..block {
                    for ic in 0..block { 
                        max_v = max_v.max(input[[r * block + ir, c * block + ic]]); 
                    }
                }
                out[[r, c]] = max_v;
            }
        }
        out
    }

    pub fn evaluate_traces_on_costmap(
        map: &Array2<f64>,
        traces: &Array2<f64>,
    ) -> (f64, f64) {
        debug_assert_eq!(map.dim(), traces.dim(), "map and traces must have same shape");

        let loss: f64 = map.iter().zip(traces.iter()).map(|(m, t)| m * t).sum();
        // let cost_fit = 1.0 / loss.max(1e-7);
        let cost_fit = -loss;
        (loss, cost_fit)
    }

    pub fn calculate_traffic_balance(g: &CoreGenome<NWSpecialization>) -> f64 {
        let n = g.nodes.len();
        if n < 2 { return 0.0; }
        
        let mut id_map = HashMap::new();
        let mut is_fixed = Vec::new();
        let mut total_terminals = 0;
        
        for (i, (&id, node)) in g.nodes.iter().enumerate() {
            id_map.insert(id, i);
            let fixed = node.data.type_ == NodeType::Fixed;
            is_fixed.push(fixed);
            if fixed { total_terminals += 1; }
        }
        
        let mut adj = vec![Vec::new(); n];
        for e in g.edges.values() {
            if !e.enabled { continue; }
            let u = id_map[&e.node1];
            let v = id_map[&e.node2];
            adj[u].push(v); 
            adj[v].push(u);
        }

        let mut tin = vec![-1; n];
        let mut low = vec![-1; n];
        let mut timer = 0;
        let mut total_severity = 0.0;
        
        fn dfs(
            u: usize, 
            p: isize, 
            timer: &mut i32, 
            tin: &mut Vec<i32>, 
            low: &mut Vec<i32>, 
            adj: &[Vec<usize>], 
            is_fixed: &[bool],
            total_terminals: u32,
            total_severity: &mut f64
        ) -> u32 {
            tin[u] = *timer; 
            low[u] = *timer; 
            *timer += 1;
            
            let mut subtree_terminals = if is_fixed[u] { 1 } else { 0 };
            
            for &v in &adj[u] {
                if v as isize == p { continue; }
                
                if tin[v] != -1 {
                    low[u] = low[u].min(tin[v]);
                } else {
                    let child_terminals = dfs(v, u as isize, timer, tin, low, adj, is_fixed, total_terminals, total_severity);
                    subtree_terminals += child_terminals;
                    
                    low[u] = low[u].min(low[v]);
                    
                    if low[v] > tin[u] {
                        let ta = child_terminals;
                        let tb = total_terminals - ta;
                        *total_severity += (ta * tb) as f64;
                    }
                }
            }
            subtree_terminals
        }
        
        for i in 0..n {
            if tin[i] == -1 {
                dfs(i, -1, &mut timer, &mut tin, &mut low, &adj, &is_fixed, total_terminals, &mut total_severity);
            }
        }
        
        -total_severity
    }

    pub fn calculate_max_edge_traffic(
        g: &CoreGenome<NWSpecialization>,
        map: &Array2<f64>
    ) -> f64 {
        let (h, w) = (map.nrows(), map.ncols());
        let n = g.nodes.len();
        if n < 2 { return 0.0; }
        
        let mut id_map = HashMap::new();
        let mut terminals = Vec::new();
        
        for (i, (&id, node)) in g.nodes.iter().enumerate() {
            id_map.insert(id, i);
            if node.data.type_ == NodeType::Fixed {
                terminals.push(i);
            }
        }
        
        if terminals.len() < 2 { return 0.0; }
        
        let mut adj = vec![Vec::new(); n];
        let mut edge_marks: HashMap<u64, usize> = HashMap::new();
        
        for e in g.edges.values() {
            if !e.enabled { continue; }
            let n1 = &g.nodes[&e.node1];
            let n2 = &g.nodes[&e.node2];
            
            let mut cost = 0.0;
            bresenham(n1.data.pos, n2.data.pos, |r, c, weight, _| {
                if r < h && c < w {
                    cost += map[[r, c]] * weight;
                }
            });
            
            let u = id_map[&e.node1];
            let v = id_map[&e.node2];
            adj[u].push((v, e.id, cost));
            adj[v].push((u, e.id, cost));
            edge_marks.insert(e.id, 0);
        }

        #[derive(PartialEq)]
        struct State { cost: f64, node: usize }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                other.cost.partial_cmp(&self.cost) // Reverse for min-heap
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        for i in 0..terminals.len() {
            let start = terminals[i];
            
            let mut dists = vec![f64::INFINITY; n];
            let mut preds = vec![None; n];
            let mut heap = std::collections::BinaryHeap::new();
            
            dists[start] = 0.0;
            heap.push(State { cost: 0.0, node: start });
            
            while let Some(State { cost, node }) = heap.pop() {
                if cost > dists[node] { continue; }
                
                for &(next_node, edge_id, edge_cost) in &adj[node] {
                    let next_cost = cost + edge_cost;
                    if next_cost < dists[next_node] {
                        dists[next_node] = next_cost;
                        preds[next_node] = Some((node, edge_id));
                        heap.push(State { cost: next_cost, node: next_node });
                    }
                }
            }
            
            for j in (i + 1)..terminals.len() {
                let mut current = terminals[j];
                while current != start {
                    if let Some((prev, edge_id)) = preds[current] {
                        if let Some(marks) = edge_marks.get_mut(&edge_id) {
                            *marks += 1;
                        }
                        current = prev;
                    } else {
                        break;
                    }
                }
            }
        }
        
        let max_marks = edge_marks.values().max().copied().unwrap_or(0);
        -(max_marks as f64)
    }

    /// Compute a single named objective for a genome that is already
    /// known to be connected.
    pub fn compute_objective(
        name: &str,
        g: &CoreGenome<NWSpecialization>,
        map: &Array2<f64>,
    ) -> f64 {
        match name {
            "cost_map" => {
                // Optimization: Instead of allocating a large HxW trace matrix 
                // and endpoint matrix for every evaluated genome,
                // we directly accumulate the cost by evaluating the nodes and edges
                // on the fly against the cost map. This eliminates allocations.
                // The canonical, slower option would be to use `compute_traces()`.
                let h = map.nrows();
                let w = map.ncols();
                let mut loss = 0.0;
                
                // 1. Node contributions
                for n in g.nodes.values() {
                    let (r, c) = (n.data.pos.0 as usize, n.data.pos.1 as usize);
                    if r < h && c < w { 
                        loss += map[[r, c]]; 
                    }
                }
        
                // 2. Edge contributions
                for e in g.edges.values() {
                    if !e.enabled { continue; }
                    let n1 = &g.nodes[&e.node1];
                    let n2 = &g.nodes[&e.node2];
                    
                    bresenham(n1.data.pos, n2.data.pos, |r, c, weight, _is_endpoint| {
                        if r < h && c < w {
                            loss += map[[r, c]] * weight;
                        }
                    });
                }
                
                // Scale loss by map size as originally done in compute_traces
                let scale = 1.0 / ((h + w) as f64);
                // Return negative loss since NEAT maximizes fitness
                -(loss * scale)
            },
            "traffic_balance" => calculate_traffic_balance(g),
            "max_edge_traffic" => calculate_max_edge_traffic(g, map),
            _ => unimplemented!("Objective '{}' is not implemented yet", name)
        }
    }

    /// Combine a list of objective values into a single scalar fitness.
    pub fn scalarize(
        values: &[f64], 
        strategy: &ScalarizationStrategy,
        utopia: &[f64],
        nadir: &[f64]
    ) -> f64 {
        match strategy {
            ScalarizationStrategy::AugmentedChebyshev { rho } => {
                let mut max_weighted_dist = f64::NEG_INFINITY;
                let mut sum_weighted_dist = 0.0;

                for i in 0..values.len() {
                    let range = utopia[i] - nadir[i];
                    let w = if range > 1e-9 { 1.0 / range } else { 1.0 };
                    
                    // Since all objectives are formulated such that higher = better (fitnesses),
                    // our Euclidean "distance from Utopia" is (utopia - val).
                    let dist = utopia[i] - values[i];
                    let weighted_dist = w * dist;

                    max_weighted_dist = max_weighted_dist.max(weighted_dist);
                    sum_weighted_dist += weighted_dist;
                }

                // Augmented Chebyshev seeks to minimize this combined distance term.
                // Since NEAT *maximizes* fitness, we must negate the final result!
                let scalar_dist = max_weighted_dist + (rho * sum_weighted_dist);
                -scalar_dist
            }
        }
    }

    pub fn evaluate_genomes_batch(
        genomes: &mut [CoreGenome<NWSpecialization>],
        map: &Array2<f64>,
        objectives: &[String],
        scalarization: Option<&ScalarizationStrategy>,
        constraints: &[String],
    ) {
        if genomes.is_empty() { return; }

        let evaluate_names: Vec<String> = if objectives.is_empty() {
            vec!["cost_map".to_string()]
        } else {
            objectives.to_vec()
        };

        let check_intersections = constraints.iter().any(|c| c == "no_intersecting_edges");

        let n_objs = evaluate_names.len();

        // Phase 1: Compute raw objectives in parallel
        let raw_results: Vec<Option<Vec<f64>>> = genomes.par_iter_mut().map(|g| {
            // Connectivity is always checked (structural invariant, also enforced at mutation time)
            let all_nodes: Vec<u64> = g.nodes.keys().copied().collect();
            let active_edges: HashMap<u64, (u64, u64)> = g.edges.iter()
                .filter(|(_, e)| e.enabled)
                .map(|(&k, e)| (k, (e.node1, e.node2)))
                .collect();

            if !check_connectivity_dict(&all_nodes, &active_edges) {
                g.fitness = f64::NEG_INFINITY;
                return None;
            }

            // Optional constraints
            if check_intersections && has_intersecting_edges(g) {
                g.fitness = f64::NEG_INFINITY;
                return None;
            }

            Some(evaluate_names.iter().map(|name| compute_objective(name, g, map)).collect())
        }).collect();

        // Phase 2: Scalarize or bypass
        if n_objs == 1 {
            for (g, res) in genomes.iter_mut().zip(raw_results.into_iter()) {
                if let Some(vals) = res {
                    g.fitness = vals[0];
                }
            }
        } else {
            let mut utopia = vec![f64::NEG_INFINITY; n_objs];
            let mut nadir = vec![f64::INFINITY; n_objs];

            for res in raw_results.iter().flatten() {
                for (i, &val) in res.iter().enumerate() {
                    utopia[i] = utopia[i].max(val);
                    nadir[i] = nadir[i].min(val);
                }
            }

            let strat = scalarization.expect("Scalarization strategy is required for multi-objective optimization");
            for (g, res) in genomes.iter_mut().zip(raw_results.into_iter()) {
                if let Some(vals) = res {
                    g.fitness = scalarize(&vals, strat, &utopia, &nadir);
                }
            }
        }
    }
}

// ============================================================================
// 4. PYTHON INTERFACE
// ============================================================================

// Structure of mod python:
//   dto       — Serde Data Transfer Objects used for Rust <-> Python genome serialization
//   wrappers  — #[pyclass] wrappers for Node, Edge, Species, Genome, Population
//   functions — #[pyfunction] entrypoints: batch ops, evolutionary ops, utilities

mod python {
    use super::*;

    // --- DTOs (Serde, genome serialization) ---

    mod dto {
        use super::*;

        #[derive(Serialize, Deserialize)]
        pub struct GenomeDTO {
            pub nodes: Vec<NodeDTO>,
            pub edges: Vec<EdgeDTO>,
        }

        #[derive(Serialize, Deserialize)]
        pub struct NodeDTO {
            #[serde(skip_serializing_if = "Option::is_none")]
            pub id: Option<u64>,
            pub data: NWNodeData,
        }

        #[derive(Serialize, Deserialize)]
        pub struct EdgeDTO {
            #[serde(skip_serializing_if = "Option::is_none")]
            pub id: Option<u64>,
            pub node1: u64,
            pub node2: u64,
            pub enabled: bool,
        }
    }

    // --- Wrappers (Node, Edge, Species, Genome, Population) ---
    #[pyclass(name = "NodeData")]
    #[derive(Clone, Debug)]
    pub struct PyNodeData {
        #[pyo3(get)]
        pub pos: (i32, i32),
        pub type_: NodeType,
    }

    #[pymethods]
    impl PyNodeData {
        #[getter]
        fn type_(&self) -> String {
            format!("{:?}", self.type_)
        }

        fn __repr__(&self) -> String {
            format!("NodeData(pos={:?}, type={:?})", self.pos, self.type_)
        }
    }

    #[pyclass(name = "Node")]
    #[derive(Clone)]
    pub struct PyNode {
        #[pyo3(get)]    pub id: u64,
        #[pyo3(get)]    pub data: PyNodeData,
    }

    #[pymethods]
    impl PyNode {
        fn __repr__(&self) -> String {
            format!(
                "Node(id={}, data={})",
                self.id, self.data.__repr__()
            )
        }
    }

    #[pyclass(name = "Edge")]
    #[derive(Clone)]
    pub struct PyEdge {
        #[pyo3(get)]    pub id: u64,
        #[pyo3(get)]    pub node1: u64,
        #[pyo3(get)]    pub node2: u64,
        #[pyo3(get)]    pub enabled: bool,
    }

    #[pymethods]
    impl PyEdge {
        fn __repr__(&self) -> String {
            format!(
                "Edge(id={}, {} -> {}, enabled={})",
                self.id, self.node1, self.node2, self.enabled
            )
        }
    }

    // --- Species Wrapper ---
    #[pyclass(name = "Species")]
    pub struct PySpecies {
        pub inner: neat_core_rs::Species<NWSpecialization>,
    }

    #[pymethods]
    impl PySpecies {
        #[getter]
        fn members(&self) -> Vec<PyGenome> {
            self.inner.members.iter()
                .map(|g| PyGenome { inner: g.clone() })
                .collect()
        }

        #[getter]
        fn representative(&self) -> PyGenome {
            PyGenome {
                inner: Arc::new(RwLock::new(self.inner.representative.clone()))
            }
        }
    }

    // --- Genome Wrapper ---
    #[pyclass(name = "Genome")]
    #[derive(Clone)]
    pub struct PyGenome {
        pub inner: Arc<RwLock<CoreGenome<NWSpecialization>>>,
    }

    #[pymethods]
    impl PyGenome {
        #[getter] 
        fn fitness(&self) -> f64 { 
            self.inner.read().unwrap().fitness 
        }
        
        #[setter] 
        fn set_fitness(&self, v: f64) { 
            self.inner.write().unwrap().fitness = v; 
        }

        fn get_structure(&self, py: Python) -> PyResult<PyObject> {
            let g = self.inner.read().unwrap();
            let dto = dto::GenomeDTO {
                nodes: g.nodes.values().map(|n| dto::NodeDTO {
                    id: Some(n.id),
                    data: n.data,
                }).collect(),
                edges: g.edges.values().map(|e| dto::EdgeDTO {
                    id: Some(e.id),
                    node1: e.node1,
                    node2: e.node2,
                    enabled: e.enabled,
                }).collect(),
            };
            Ok(pythonize(py, &dto)?.into())
        }

        #[getter]
        fn nodes(&self) -> HashMap<u64, PyNode> {
            self.inner.read().unwrap().nodes.iter().map(|(&k, v)| {
                (k, PyNode {
                    id: v.id,
                    data: PyNodeData {
                        pos: v.data.pos,
                        type_: v.data.type_,
                    }
                })
            }).collect()
        }

        #[getter]
        fn edges(&self) -> HashMap<u64, PyEdge> {
            self.inner.read().unwrap().edges.iter().map(|(&k, v)| {
                (k, PyEdge {
                    id: v.id,
                    node1: v.node1,
                    node2: v.node2,
                    enabled: v.enabled,
                })
            }).collect()
        }
    }

    // --- Population Wrapper ---
    #[pyclass(name = "Population")]
    pub struct PyPopulation {
        pub core: Arc<RwLock<CorePopulation<NWSpecialization>>>,
    }

    #[pymethods]
    impl PyPopulation {
        #[new]
        #[pyo3(signature = (config, start_genome=None))]
        fn new(_py: Python, config: &Bound<PyAny>, start_genome: Option<PyGenome>) -> PyResult<Self> {
            let rust_config: CoreNEATConfig<NWConfig> = depythonize(config)?;
            let spec = NWSpecialization { config: rust_config.special.clone() };
            let mut pop = CorePopulation::new(rust_config, spec);
            pop.initialize(start_genome.map(|g| g.inner.read().unwrap().clone()).as_ref());
            Ok(Self { core: Arc::new(RwLock::new(pop)) })
        }

        fn get_initial_genomes(&self) -> Vec<PyGenome> {
            self.core.read().unwrap().members.iter()
                .map(|g| PyGenome { inner: g.clone() })
                .collect()
        }

        #[getter]
        fn members(&self) -> Vec<PyGenome> {
            self.core.read().unwrap().members.iter()
                .map(|g| PyGenome { inner: g.clone() })
                .collect()
        }

        #[getter]
        fn species(&self) -> Vec<PySpecies> {
            self.core.read().unwrap().species.iter()
                .map(|s| PySpecies { inner: s.clone() })
                .collect()
        }

        fn get_stats(&self) -> (f64, f64, usize, Vec<f64>) {
            let pop = self.core.read().unwrap();
            let max = pop.members.iter()
                .map(|g| g.read().unwrap().fitness)
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));
            let sum: f64 = pop.members.iter()
                .map(|g| g.read().unwrap().fitness)
                .filter(|&f| f > f64::NEG_INFINITY)
                .sum();
            let num_valid = pop.members.iter()
                .filter(|g| g.read().unwrap().fitness > f64::NEG_INFINITY)
                .count();
            let avg = if num_valid > 0 { sum / num_valid as f64 } else { 0.0 };

            let species_fits: Vec<f64> = pop.species.iter()
                .map(|s| s.members.iter()
                    .map(|g| g.read().unwrap().fitness)
                    .fold(f64::NEG_INFINITY, |a, b| a.max(b))
                ).collect();
            (if max == f64::NEG_INFINITY { 0.0 } else { max }, avg, pop.species.len(), species_fits)
        }

        fn new_node_id(&self) -> u64 {
            self.core.read().unwrap().state.new_node_id()
        }

        fn get_edge_id(&self, n1: u64, n2: u64) -> u64 {
            let pop = self.core.read().unwrap();
            pop.state.get_edge_id(&pop.specialization, n1, n2)
        }

        #[getter]
        fn current_species_threshold(&self) -> f64 {
            self.core.read().unwrap().current_species_threshold
        }

        /// Creates a genome from a Python dict structure, with safe ID management.
        /// 
        /// Example:
        ///     # Create a custom genome with auto-generated IDs:
        ///     structure = {
        ///         "nodes": [
        ///             {"data": {"pos": (5, 5), "type": "Fixed"}},
        ///             {"data": {"pos": (10, 10), "type": "Flexible"}},
        ///         ],
        ///         "edges": [
        ///             {"node1": n1_id, "node2": n2_id, "enabled": True}
        ///         ]
        ///     }
        ///     genome = pop.create_genome(structure)
        ///     
        ///     # Or manually manage IDs:
        ///     n1_id = pop.new_node_id()
        ///     n2_id = pop.new_node_id()
        ///     e_id = pop.get_edge_id(n1_id, n2_id)
        ///     structure = {
        ///         "nodes": [
        ///             {"id": n1_id, "data": {"pos": (5, 5), "type": "Fixed"}},
        ///             {"id": n2_id, "data": {"pos": (10, 10), "type": "Flexible"}},
        ///         ],
        ///         "edges": [
        ///             {"id": e_id, "node1": n1_id, "node2": n2_id, "enabled": True}
        ///         ]
        ///     }
        fn create_genome(&self, _py: Python, structure: &Bound<PyDict>) -> PyResult<PyGenome> {
            let dto: dto::GenomeDTO = depythonize(structure)?;
            let pop = self.core.write().unwrap();
            
            let mut nodes = HashMap::new();
            let mut edges = HashMap::new();
            
            // 1. Update counters to accommodate imported IDs
            // We scan ahead to ensure the global counters are high enough to cover all imported IDs.
            // This prevents collisions if you mix explicit IDs with auto-generated ones later.
            let mut max_node_id = 0;
            let mut max_edge_id = 0;
            
            for n in &dto.nodes {
                if let Some(id) = n.id { max_node_id = max_node_id.max(id); }
            }
            for e in &dto.edges {
                if let Some(id) = e.id { max_edge_id = max_edge_id.max(id); }
            }

            {
                let mut g = pop.state.node_id.lock().unwrap();
                *g = (*g).max(max_node_id);
            }
            {
                let mut g = pop.state.edge_id.lock().unwrap();
                *g = (*g).max(max_edge_id);
            }
            
            // 2. Create Nodes
            for n in dto.nodes {
                // If ID is missing, generate a new one (which is now guaranteed > max_node_id)
                let id = n.id.unwrap_or_else(|| pop.state.new_node_id());
                nodes.insert(id, CoreNode::new(id, Some(n.data)));
            }

            // 3. Create Edges
            for e in dto.edges {
                let id = if let Some(existing_id) = e.id {
                    // Register gene mapping so future lookups for this pair return this ID
                    let key = pop.specialization.normalize_edge(e.node1, e.node2);
                    pop.state.edge_genes.lock().unwrap().insert(key, existing_id);
                    existing_id
                } else {
                    pop.state.get_edge_id(&pop.specialization, e.node1, e.node2)
                };
                edges.insert(id, CoreEdge::new(e.node1, e.node2, id, Some(e.enabled), None));
            }

            let genome = CoreGenome::new(
                pop.config.clone(),
                pop.specialization.clone(),
                pop.state.clone(),
                Some(edges),
                Some(nodes)
            );
            Ok(PyGenome { inner: Arc::new(RwLock::new(genome)) })
        }

        /// Initializes a new genome using the specialization's initialize_genome method.
        /// Returns a fresh genome with the default structure defined by the graph_init setting.
        fn initialize_genome(&self, _py: Python) -> PyGenome {
            let pop = self.core.read().unwrap();
            let (nodes_vec, edges_vec) = pop.specialization.initialize_genome(&*pop);
            
            let mut nodes = HashMap::new();
            let mut edges = HashMap::new();
            
            for node in nodes_vec {
                nodes.insert(node.id, node);
            }
            
            for edge in edges_vec {
                edges.insert(edge.id, edge);
            }
            
            let genome = CoreGenome::new(
                pop.config.clone(),
                pop.specialization.clone(),
                pop.state.clone(),
                Some(edges),
                Some(nodes)
            );
            PyGenome { inner: Arc::new(RwLock::new(genome)) }
        }
    }

    // --- Evolutionary operations (batch eval, mutation, reproduction, selection) ---

    #[pyfunction]
    #[pyo3(signature = (genomes, cost_map, objectives=vec![], scalarization=None, constraints=None))]
    pub fn evaluate_genomes(
        py: Python,
        genomes: Vec<PyGenome>,
        cost_map: PyReadonlyArray2<f64>,
        objectives: Vec<String>,
        scalarization: Option<Bound<'_, PyAny>>,
        constraints: Option<Vec<String>>,
    ) {
        let map = cost_map.as_array().to_owned();
        let scalar_strat: Option<ScalarizationStrategy> = scalarization
            .map(|s| depythonize(&s).expect("Failed to deserialize scalarization strategy"));
        let constraints = constraints.unwrap_or_default();

        let mut cloned_genomes: Vec<_> = genomes.iter().map(|g| g.inner.read().unwrap().clone()).collect();
        let evaluated_genomes = py.allow_threads(move || {
            evaluation::evaluate_genomes_batch(&mut cloned_genomes, &map, &objectives, scalar_strat.as_ref(), &constraints);
            cloned_genomes
        });

        for (py_genome, evaluated_genome) in genomes.iter().zip(evaluated_genomes.into_iter()) {
            *py_genome.inner.write().unwrap() = evaluated_genome;
        }
    }

    #[pyfunction]
    pub fn mutate_genomes(py: Python, genomes: Vec<PyGenome>, pop: &PyPopulation) {
        let (conf, spec) = {
            let p = pop.core.read().unwrap();
            (p.config.clone(), p.specialization.clone())
        };
        py.allow_threads(move || {
            genomes.par_iter().for_each(|g| {
                let mut inner = g.inner.write().unwrap();
                inner.config = conf.clone();
                inner.specialization = spec.clone();
                inner.mutate(&mut rand::thread_rng());
            });
        });
    }

    #[pyfunction]
    pub fn make_offspring(
        py: Python, 
        pop: &PyPopulation, 
        count: usize
    ) -> (Vec<PyGenome>, Vec<PyGenome>, Vec<PyGenome>) {
        let core = pop.core.clone();
        let (e, s, c) = py.allow_threads(move || {
            let mut p = core.write().unwrap();
            p.config.population_size = count;
            p.reproduce(&mut rand::thread_rng())
        });
        (
            e.into_iter().map(|g| PyGenome { inner: g }).collect(),
            s.into_iter().map(|g| PyGenome { inner: g }).collect(),
            c.into_iter().map(|g| PyGenome { inner: g }).collect()
        )
    }

    #[pyfunction]
    pub fn select(
        py: Python, 
        pop: &PyPopulation, 
        elites: Vec<PyGenome>, 
        species_elites: Vec<PyGenome>,
        children: Vec<PyGenome>
    ) {
        let core = pop.core.clone();
        let (e, s, c) = (
            elites.into_iter().map(|g| g.inner).collect(),
            species_elites.into_iter().map(|g| g.inner).collect(),
            children.into_iter().map(|g| g.inner).collect()
        );
        py.allow_threads(move || {
            core.write().unwrap().select(e, s, c, &mut rand::thread_rng());
        });
    }

    // --- Utility functions (inspection, scoring, compatibility) ---

    #[pyfunction]
    pub fn get_genome_mask<'py>(
        py: Python<'py>,
        genome: &PyGenome,
        shape: (usize, usize)
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        Ok(evaluation::compute_traces(
            shape.0, 
            shape.1, 
            &genome.inner.read().unwrap()
        ).to_pyarray(py))
    }

    #[pyfunction]
    pub fn get_trace_difference<'py>(
        py: Python<'py>,
        g1: &PyGenome,
        g2: &PyGenome,
        shape: (usize, usize),
        sigma: usize
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let t1 = evaluation::compute_traces(shape.0, shape.1, &g1.inner.read().unwrap());
        let t2 = evaluation::compute_traces(shape.0, shape.1, &g2.inner.read().unwrap());
        
        let s1 = evaluation::max_pool(&t1, sigma);
        let s2 = evaluation::max_pool(&t2, sigma);

        let mut diff = Array2::<f64>::zeros(s1.dim());
        for ((r, c), &v1) in s1.indexed_iter() {
            diff[[r, c]] = (v1 - s2[[r, c]]).abs();
        }
        Ok(diff.to_pyarray(py))
    }

    #[pyfunction]
    pub fn genome_compatibility(g1: &PyGenome, g2: &PyGenome) -> f64 {
        neat_core_rs::calculate_compatibility(
            &*g1.inner.read().unwrap(),
            &*g2.inner.read().unwrap()
        )
    }

    /// Compute a single named objective for a genome.
    /// Supported names: `"cost_map"`, `"traffic_balance"`, `"max_edge_traffic"`.
    #[pyfunction]
    pub fn compute_objective(
        py: Python,
        genome: &PyGenome,
        objective: &str,
        cost_map: PyReadonlyArray2<f64>,
    ) -> f64 {
        let map = cost_map.as_array().to_owned();
        let g = genome.inner.read().unwrap().clone();
        py.allow_threads(move || evaluation::compute_objective(objective, &g, &map))
    }
}

#[pymodule]
fn neatwork_lib(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<python::PyGenome>()?;
    m.add_class::<python::PyPopulation>()?;
    m.add_class::<python::PySpecies>()?;  // Register Species
    m.add_class::<python::PyNode>()?;
    m.add_class::<python::PyNodeData>()?;
    m.add_class::<python::PyEdge>()?;
    
    m.add_function(wrap_pyfunction!(python::evaluate_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(python::mutate_genomes, m)?)?;
    m.add_function(wrap_pyfunction!(python::make_offspring, m)?)?;
    m.add_function(wrap_pyfunction!(python::select, m)?)?;
    m.add_function(wrap_pyfunction!(python::get_genome_mask, m)?)?;
    m.add_function(wrap_pyfunction!(python::get_trace_difference, m)?)?;
    m.add_function(wrap_pyfunction!(python::genome_compatibility, m)?)?;
    m.add_function(wrap_pyfunction!(python::compute_objective, m)?)?;
    Ok(())
}
