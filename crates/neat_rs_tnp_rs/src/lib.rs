use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyReadonlyArray2, ToPyArray};
use ndarray::Array2;
use neat_rs_core::{
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
use std::cmp::min;

// ============================================================================
// 1. DOMAIN TYPES (Unified Rust + Python + Serde)
// ============================================================================

mod types {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub enum NodeType { Fixed, Flexible }
    impl Default for NodeType { fn default() -> Self { NodeType::Flexible } }

    #[derive(Clone, Debug, Default, Copy, Serialize, Deserialize)]
    pub struct TNPNodeData {
        pub pos: (i32, i32),
        #[serde(rename = "type")]
        pub type_: NodeType,
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    pub struct TNPEdgeData;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub enum EvaluationStrategy { SingleObjective, MultiObjectiveBridge }
    impl Default for EvaluationStrategy { fn default() -> Self { Self::SingleObjective } }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub enum GraphInit { FCN, Grid(u32, u32) }
    impl Default for GraphInit { fn default() -> Self { Self::FCN } }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct TNPConfig {
        pub grid_size: (usize, usize),
        pub fixed_nodes: Vec<(i32, i32)>,
        pub move_node_mutation_prob: f64,
        pub move_node_mutation_sigma: f64,
        pub evaluation_strategy: EvaluationStrategy,
        pub graph_init: GraphInit,
        pub compatibility_sigma: usize,
        pub c3: f64,
    }

    #[derive(Clone, Debug)]
    pub struct TNPSpecialization { pub config: TNPConfig }
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
            nodes: &HashMap<u64, CoreNode<TNPNodeData>>, 
            ignore: Option<u64>
        ) -> bool {
            nodes.values().any(|n| n.data.pos == pos && Some(n.id) != ignore)
        }

        pub fn find_free_pos(
            &self, 
            center: (i32, i32), 
            nodes: &HashMap<u64, CoreNode<TNPNodeData>>, 
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

    impl Specialization for TNPSpecialization {
        type NodeData = TNPNodeData;
        type EdgeData = TNPEdgeData;
        type Config = TNPConfig;

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
                    Some(TNPNodeData { pos, type_: NodeType::Fixed })
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
                                Some(TNPNodeData { pos, type_: NodeType::Flexible })
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
                new_node.data = TNPNodeData { pos, type_: NodeType::Flexible };
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

    pub fn compute_traces(
        h: usize, 
        w: usize, 
        g: &CoreGenome<TNPSpecialization>
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
            
            let (r0, c0) = (n1.data.pos.0 as f64, n1.data.pos.1 as f64);
            let (r1, c1) = (n2.data.pos.0 as f64, n2.data.pos.1 as f64);
            
            let dist = ((r0 - r1).powi(2) + (c0 - c1).powi(2)).sqrt();
            if dist < 1e-5 { continue; }
            
            let n_steps = (r0 - r1).abs().max((c0 - c1).abs()).ceil() as usize + 1;
            if n_steps < 2 { continue; }
            
            // Weight respects euclidean distance
            let edge_vec = ((r1 - r0).abs(), (c1 - c0).abs());
            let (a, b) = (edge_vec.0.max(edge_vec.1), edge_vec.0.min(edge_vec.1));
            let c_div_a = if a > 0.0 { (a.powi(2) + b.powi(2)).sqrt() / a } else { 1.0 };
            let weight = c_div_a;
            
            for i in 0..n_steps {
                let t = i as f64 / (n_steps as f64 - 1.0);
                let r = (r0 * (1.0 - t) + r1 * t).round() as usize;
                let c = (c0 * (1.0 - t) + c1 * t).round() as usize;
                
                if r < h && c < w {
                    if i == 0 || i == n_steps - 1 { 
                        endpoints[[r, c]] += weight; 
                    } else { 
                        traces[[r, c]] += weight; 
                    }
                }
            }
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

    pub fn calculate_bridge_share(g: &CoreGenome<TNPSpecialization>) -> f64 {
        let n = g.nodes.len();
        if n < 2 { return 0.0; }
        
        let mut id_map = HashMap::new();
        let mut coords = Vec::new();
        for (i, (&id, node)) in g.nodes.iter().enumerate() {
            id_map.insert(id, i);
            coords.push(node.data.pos);
        }
        
        let mut adj = vec![Vec::new(); n];
        let mut total_len = 0.0;
        for e in g.edges.values() {
            if !e.enabled { continue; }
            let (u, v) = (id_map[&e.node1], id_map[&e.node2]);
            adj[u].push(v); 
            adj[v].push(u);
            let (p1, p2) = (coords[u], coords[v]);
            total_len += (((p1.0 - p2.0).pow(2) + (p1.1 - p2.1).pow(2)) as f64).sqrt();
        }
        if total_len < 1e-9 { return 0.0; }

        let mut tin = vec![-1; n];
        let mut low = vec![-1; n];
        let mut timer = 0;
        let mut bridge_len = 0.0;
        
        fn dfs(
            u: usize, 
            p: isize, 
            timer: &mut i32, 
            tin: &mut Vec<i32>, 
            low: &mut Vec<i32>, 
            adj: &Vec<Vec<usize>>, 
            coords: &Vec<(i32, i32)>, 
            bridge_len: &mut f64
        ) {
            tin[u] = *timer; 
            low[u] = *timer; 
            *timer += 1;
            for &v in &adj[u] {
                if v as isize == p { continue; }
                if tin[v] != -1 {
                    low[u] = min(low[u], tin[v]);
                } else {
                    dfs(v, u as isize, timer, tin, low, adj, coords, bridge_len);
                    low[u] = min(low[u], low[v]);
                    if low[v] > tin[u] {
                        let (p1, p2) = (coords[u], coords[v]);
                        *bridge_len += (((p1.0 - p2.0).pow(2) + (p1.1 - p2.1).pow(2)) as f64).sqrt();
                    }
                }
            }
        }
        
        dfs(0, -1, &mut timer, &mut tin, &mut low, &adj, &coords, &mut bridge_len);
        bridge_len / total_len
    }

    pub fn evaluate_genome(g: &mut CoreGenome<TNPSpecialization>, map: &Array2<f64>) {
        let all_nodes: Vec<u64> = g.nodes.keys().copied().collect();
        let active_edges: HashMap<u64, (u64, u64)> = g.edges.iter()
            .filter(|(_, e)| e.enabled)
            .map(|(&k, e)| (k, (e.node1, e.node2)))
            .collect();
        
        if !check_connectivity_dict(&all_nodes, &active_edges) { 
            g.fitness = 0.0; 
            if g.config.special.evaluation_strategy == EvaluationStrategy::MultiObjectiveBridge {
                g.objectives = Some(vec![0.0, 0.0]);
            }
            return; 
        }

        let traces = compute_traces(map.nrows(), map.ncols(), g);
        let loss: f64 = map.iter().zip(traces.iter()).map(|(m, t)| m * t).sum();
        let cost_fit = 1.0 / loss.max(1e-7);
        
        g.fitness = cost_fit;
        if g.config.special.evaluation_strategy == EvaluationStrategy::MultiObjectiveBridge {
            let robustness = 1.0 - calculate_bridge_share(g);
            g.objectives = Some(vec![cost_fit, robustness]);
        }
    }
}

// ============================================================================
// 4. PYTHON INTERFACE
// ============================================================================

mod python {
    use super::*;

    // --- Serde DTO for IO (Single unified type) ---
    // DTO = Data Transfer Object: a plain data-only struct used for serialization/deserialization
    // and moving data across boundaries (e.g., Rust <-> Python, or Rust <-> JSON), without
    // embedding business logic.
    #[derive(Serialize, Deserialize)]
    struct GenomeDTO {
        nodes: Vec<NodeDTO>,
        edges: Vec<EdgeDTO>,
    }

    #[derive(Serialize, Deserialize)]
    struct NodeDTO {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<u64>,
        data: TNPNodeData,
    }

    #[derive(Serialize, Deserialize)]
    struct EdgeDTO {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<u64>,
        node1: u64,
        node2: u64,
        enabled: bool,
    }

    // --- Node/Edge Inspection Wrappers ---
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
        pub inner: neat_rs_core::Species<TNPSpecialization>,
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
        pub inner: Arc<RwLock<CoreGenome<TNPSpecialization>>>,
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

        #[getter]
        fn objectives(&self) -> Option<Vec<f64>> {
            self.inner.read().unwrap().objectives.clone()
        }

        #[setter]
        fn set_objectives(&self, v: Option<Vec<f64>>) {
            self.inner.write().unwrap().objectives = v;
        }

        fn get_structure(&self, py: Python) -> PyResult<PyObject> {
            let g = self.inner.read().unwrap();
            let dto = GenomeDTO {
                nodes: g.nodes.values().map(|n| NodeDTO {
                    id: Some(n.id),
                    data: n.data,
                }).collect(),
                edges: g.edges.values().map(|e| EdgeDTO {
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
        pub core: Arc<RwLock<CorePopulation<TNPSpecialization>>>,
    }

    #[pymethods]
    impl PyPopulation {
        #[new]
        #[pyo3(signature = (config, start_genome=None))]
        fn new(_py: Python, config: &Bound<PyAny>, start_genome: Option<PyGenome>) -> PyResult<Self> {
            let rust_config: CoreNEATConfig<TNPConfig> = depythonize(config)?;
            let spec = TNPSpecialization { config: rust_config.special.clone() };
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
                .fold(0.0f64, |a, b| a.max(b));
            let sum: f64 = pop.members.iter()
                .map(|g| g.read().unwrap().fitness)
                .sum();
            let species_fits: Vec<f64> = pop.species.iter()
                .map(|s| s.members.iter()
                    .map(|g| g.read().unwrap().fitness)
                    .fold(0.0f64, |a, b| a.max(b))
                ).collect();
            let avg = sum / pop.members.len().max(1) as f64;
            (max, avg, pop.species.len(), species_fits)
        }

        fn augment_diversity(&self, py: Python, genomes: Vec<PyGenome>) {
            let core = self.core.clone();
            let refs: Vec<_> = genomes.into_iter().map(|g| g.inner).collect();
            py.allow_threads(move || {
                core.read().unwrap().augment_diversity(&refs);
            });
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
            let dto: GenomeDTO = depythonize(structure)?;
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

    // --- Batch Functions ---
    #[pyfunction]
    pub fn evaluate_genomes(py: Python, genomes: Vec<PyGenome>, cost_map: PyReadonlyArray2<f64>) {
        let map = cost_map.as_array().to_owned();
        py.allow_threads(move || {
            genomes.par_iter().for_each(|g| {
                evaluation::evaluate_genome(&mut g.inner.write().unwrap(), &map);
            });
        });
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

    // --- Utility Functions ---
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
        neat_rs_core::calculate_compatibility(
            &*g1.inner.read().unwrap(),
            &*g2.inner.read().unwrap()
        )
    }

    #[pyfunction]
    pub fn get_pareto_front(genomes: Vec<PyGenome>) -> Vec<PyGenome> {
        let mut front = Vec::new();
        // Pre-fetch objectives to avoid locking in the loop
        let objs: Vec<Option<Vec<f64>>> = genomes.iter()
            .map(|g| g.inner.read().unwrap().objectives.clone())
            .collect();

        for (i, g1) in genomes.iter().enumerate() {
            if let Some(o1) = &objs[i] {
                let mut dominated = false;
                for (j, _) in genomes.iter().enumerate() {
                    if i == j { continue; }
                    if let Some(o2) = &objs[j] {
                        // Check if o2 dominates o1 (Maximization)
                        // A dominates B if A >= B for all and A > B for at least one
                        let mut strictly_better = false;
                        let mut equal_or_better = true;
                        
                        for k in 0..o1.len() {
                            if o2[k] < o1[k] {
                                equal_or_better = false;
                                break;
                            }
                            if o2[k] > o1[k] {
                                strictly_better = true;
                            }
                        }
                        
                        if equal_or_better && strictly_better {
                            dominated = true;
                            break;
                        }
                    }
                }
                if !dominated {
                    front.push(g1.clone());
                }
            }
        }
        front
    }

    #[pyfunction]
    pub fn process_multi_objective(
        py: Python, 
        pop: &PyPopulation, 
        genomes: Vec<PyGenome>
    ) {
        let core = pop.core.clone();
        let mut refs: Vec<_> = genomes.into_iter().map(|g| g.inner).collect();
        py.allow_threads(move || {
            core.read().unwrap().process_multi_objective(&mut refs);
        });
    }
}

#[pymodule]
fn neat_rs_tnp_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(python::get_pareto_front, m)?)?;
    m.add_function(wrap_pyfunction!(python::process_multi_objective, m)?)?;
    Ok(())
}
