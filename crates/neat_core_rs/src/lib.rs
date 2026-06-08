use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use rand::prelude::*;
use std::fmt::Debug;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

// --- Type Alias ---
pub type GenomeRef<S> = Arc<RwLock<Genome<S>>>;

// --- Trait Definition ---

/// The Specialization trait allows injecting problem-specific logic (like TNP geometry)
/// into the generic NEAT core.
pub trait Specialization: Clone + Send + Sync + 'static {
    type NodeData: Clone + Debug + Send + Sync + Default;
    type EdgeData: Clone + Debug + Send + Sync + Default;
    type Config: Clone + Debug + Send + Sync;
    
    // Factory method for initial graph
    fn initialize_genome(&self, pop: &Population<Self>) -> (Vec<Node<Self::NodeData>>, Vec<Edge<Self::EdgeData>>);
    
    // Canonical edge indexing (e.g., sort node IDs)
    fn normalize_edge(&self, n1: u64, n2: u64) -> (u64, u64);
    
    // Mutation hooks - return true if mutation is allowed/successful
    fn add_node(
        &self, 
        new_node: &mut Node<Self::NodeData>, 
        old_edge: &Edge<Self::EdgeData>, 
        e1: &mut Edge<Self::EdgeData>, 
        e2: &mut Edge<Self::EdgeData>, 
        nodes: &HashMap<u64, Node<Self::NodeData>>, 
        edges: &HashMap<u64, Edge<Self::EdgeData>>,
        rng: &mut impl Rng
    ) -> bool;
    
    fn add_edge(
        &self, 
        edge: &mut Edge<Self::EdgeData>, 
        nodes: &HashMap<u64, Node<Self::NodeData>>, 
        edges: &HashMap<u64, Edge<Self::EdgeData>>,
        rng: &mut impl Rng
    ) -> bool;
    
    fn remove_edge(
        &self, 
        edge: &Edge<Self::EdgeData>, 
        nodes: &HashMap<u64, Node<Self::NodeData>>, 
        edges: &HashMap<u64, Edge<Self::EdgeData>>,
        rng: &mut impl Rng
    ) -> bool;
    
    fn remove_node(
        &self, 
        node: &Node<Self::NodeData>, 
        edge1: &Edge<Self::EdgeData>, 
        edge2: &Edge<Self::EdgeData>, 
        new_edge: &Edge<Self::EdgeData>,
        nodes: &HashMap<u64, Node<Self::NodeData>>, 
        edges: &HashMap<u64, Edge<Self::EdgeData>>,
        rng: &mut impl Rng
    ) -> bool;
    
    fn remove_node_single(
        &self, 
        node: &Node<Self::NodeData>, 
        edge: &Edge<Self::EdgeData>, 
        nodes: &HashMap<u64, Node<Self::NodeData>>, 
        edges: &HashMap<u64, Edge<Self::EdgeData>>,
        rng: &mut impl Rng
    ) -> bool;
    
    fn mutate_structure(
        &self, 
        nodes: &mut HashMap<u64, Node<Self::NodeData>>, 
        edges: &mut HashMap<u64, Edge<Self::EdgeData>>,
        rng: &mut impl Rng
    );
    
    fn compatibility(
        &self, 
        g1: &Genome<Self>, 
        g2: &Genome<Self>,
        matching_genes: &[u64]
    ) -> f64;
}

// --- Enums ---

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum MatingStrategy {
    NEATSpeciation,
    GlobalTournament,
}

// --- Config ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NEATConfig<SpecConfig = ()> {
    pub population_size: usize,
    pub add_node_mutation_prob: f64,
    pub add_edge_mutation_prob: f64,
    pub remove_edge_mutation_prob: f64,
    pub remove_node_mutation_prob: f64,
    pub remove_node_single_mutation_prob: f64,
    pub num_elites_species: usize,
    pub num_elites_global: usize,
    pub selection_share: f64,
    pub tournament_size: usize,
    pub species_threshold: f64,
    pub target_species: Option<usize>,
    pub mating_strategy: MatingStrategy,
    pub crossover_enabled: bool,
    pub c1: f64,
    pub c2: f64,
    pub special: SpecConfig,  // Single source of truth for specialization config
}

impl Default for NEATConfig<()> {
    fn default() -> Self {
        Self {
            population_size: 100,
            add_node_mutation_prob: 0.4,
            add_edge_mutation_prob: 0.4,
            remove_edge_mutation_prob: 0.1,
            remove_node_mutation_prob: 0.1,
            remove_node_single_mutation_prob: 0.1,
            num_elites_species: 1,
            num_elites_global: 1,
            selection_share: 0.2,
            tournament_size: 2,
            species_threshold: 3.0,
            target_species: None,
            mating_strategy: MatingStrategy::NEATSpeciation,
            crossover_enabled: true,
            c1: 1.0,
            c2: 1.0,
            special: (),  // Default empty specialization config
        }
    }
}

// --- Genes ---

#[derive(Clone, Debug)]
pub struct Node<D> {
    pub id: u64,
    pub data: D,
}

impl<D: Default> Node<D> {
    pub fn new(id: u64, data: Option<D>) -> Self {
        Node {
            id,
            data: data.unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge<D> {
    pub node1: u64,
    pub node2: u64,
    pub id: u64,
    pub enabled: bool,
    pub data: D,
}

impl<D: Default> Edge<D> {
    pub fn new(node1: u64, node2: u64, id: u64, enabled: Option<bool>, data: Option<D>) -> Self {
        Edge {
            node1,
            node2,
            id,
            enabled: enabled.unwrap_or(true),
            data: data.unwrap_or_default(),
        }
    }
}

// --- Genome ---

#[derive(Clone)]
pub struct Genome<S: Specialization> {
    pub config: NEATConfig<S::Config>,
    pub specialization: S,
    pub population_state: Arc<PopulationState<S>>, 
    
    // Evaluation state
    pub fitness: f64,
    pub adj_fitness: Option<f64>,
    pub is_elite: bool, 
    
    // Graph state
    pub nodes: HashMap<u64, Node<S::NodeData>>,
    pub edges: HashMap<u64, Edge<S::EdgeData>>,
}

impl<S: Specialization> Genome<S> {
    pub fn new(
        config: NEATConfig<S::Config>,
        specialization: S,
        population_state: Arc<PopulationState<S>>,
        edges: Option<HashMap<u64, Edge<S::EdgeData>>>,
        nodes: Option<HashMap<u64, Node<S::NodeData>>>,
    ) -> Self {
        Genome {
            config,
            specialization,
            population_state,
            fitness: 0.0,
            adj_fitness: None,
            is_elite: false,
            nodes: nodes.map(|m| m.into_iter().collect()).unwrap_or_default(),
            edges: edges.map(|m| m.into_iter().collect()).unwrap_or_default(),
        }
    }
    
    pub fn sorted_edges(&self) -> Vec<&Edge<S::EdgeData>> {
        let mut edges: Vec<&Edge<S::EdgeData>> = self.edges.values().collect();
        edges.sort_by_key(|e| e.id);
        edges
    }
    
    pub fn get_node_degree(&self, node_id: u64) -> usize {
        self.edges.values()
        .filter(|e| e.enabled && (e.node1 == node_id || e.node2 == node_id))
        .count()
    }
    
    pub fn mutate(&mut self, rng: &mut impl Rng) {
        if self.is_elite { return; }
        
        // 1. Add Node
        if rng.gen_bool(self.config.add_node_mutation_prob) {
            let enabled_edges: Vec<_> = self.edges.values().filter(|e| e.enabled).collect();
            if let Some(edge_to_split) = enabled_edges.choose(rng) {
                let edge_clone = (*edge_to_split).clone();
                self.perform_add_node(&edge_clone, rng);
            }
        }
        
        // 2. Add Edge
        if rng.gen_bool(self.config.add_edge_mutation_prob) && self.nodes.len() >= 2 {
            let node_ids: Vec<u64> = self.nodes.keys().cloned().collect();
            // Try 20 times to find a valid non-existing edge
            for _ in 0..20 {
                let id1 = *node_ids.choose(rng).unwrap();
                let id2 = *node_ids.choose(rng).unwrap();
                if id1 == id2 { continue; }
                if self.perform_add_edge(id1, id2, rng) { break; }
            }
        }
        
        // 3. Remove Edge
        if rng.gen_bool(self.config.remove_edge_mutation_prob) && !self.edges.is_empty() {
            let ids: Vec<u64> = self.edges.keys().cloned().collect();
            let id = *ids.choose(rng).unwrap();
            self.perform_remove_edge(id, rng);
        }
        
        // 4. Remove Node (degree 2)
        if rng.gen_bool(self.config.remove_node_mutation_prob) && !self.nodes.is_empty() {
            let node_ids: Vec<u64> = self.nodes.keys()
            .filter(|&&id| self.get_node_degree(id) == 2)
            .cloned().collect();
            if let Some(&id) = node_ids.choose(rng) {
                self.perform_remove_node(id, rng);
            }
        }
        
        // 5. Remove Node Single (degree 1)
        if rng.gen_bool(self.config.remove_node_single_mutation_prob) && !self.nodes.is_empty() {
            let node_ids: Vec<u64> = self.nodes.keys()
            .filter(|&&id| self.get_node_degree(id) == 1)
            .cloned().collect();
            if let Some(&id) = node_ids.choose(rng) {
                self.perform_remove_node_single(id, rng);
            }
        }
        
        // 6. Specialization Mutation (e.g. geometric moves)
        self.specialization.mutate_structure(&mut self.nodes, &mut self.edges, rng);
    }
    
    fn perform_add_node(&mut self, old_edge: &Edge<S::EdgeData>, rng: &mut impl Rng) {
        let new_node_id = self.population_state.new_node_id();
        let mut new_node = Node::new(new_node_id, None);
        
        let edge1_id = self.population_state.get_edge_id(&self.specialization, old_edge.node1, new_node_id);
        let mut edge1 = Edge::new(old_edge.node1, new_node_id, edge1_id, None, None);
        
        let edge2_id = self.population_state.get_edge_id(&self.specialization, new_node_id, old_edge.node2);
        let mut edge2 = Edge::new(new_node_id, old_edge.node2, edge2_id, None, None);
        
        if self.specialization.add_node(&mut new_node, old_edge, &mut edge1, &mut edge2, &self.nodes, &self.edges, rng) {
            self.nodes.insert(new_node_id, new_node);
            self.edges.insert(edge1_id, edge1);
            self.edges.insert(edge2_id, edge2);
            if let Some(e) = self.edges.get_mut(&old_edge.id) { e.enabled = false; }
        }
    }
    
    fn perform_add_edge(&mut self, n1: u64, n2: u64, rng: &mut impl Rng) -> bool {
        let edge_id = self.population_state.get_edge_id(&self.specialization, n1, n2);
        if self.edges.contains_key(&edge_id) { return false; }
        
        let mut new_edge = Edge::new(n1, n2, edge_id, None, None);
        if self.specialization.add_edge(&mut new_edge, &self.nodes, &self.edges, rng) {
            self.edges.insert(edge_id, new_edge);
            return true;
        }
        false
    }
    
    fn perform_remove_edge(&mut self, edge_id: u64, rng: &mut impl Rng) {
        if let Some(edge) = self.edges.get(&edge_id) {
            if self.specialization.remove_edge(edge, &self.nodes, &self.edges, rng) {
                if let Some(e) = self.edges.get_mut(&edge_id) { e.enabled = false; }
            }
        }
    }
    
    fn perform_remove_node(&mut self, node_id: u64, rng: &mut impl Rng) {
        // Collect edges connected to this node
        let connected: Vec<u64> = self.edges.values()
        .filter(|e| (e.node1 == node_id || e.node2 == node_id))
        .map(|e| e.id).collect();
        let enabled: Vec<Edge<S::EdgeData>> = connected.iter()
        .filter_map(|&id| self.edges.get(&id).cloned())
        .filter(|e| e.enabled)
        .collect();
        
        if enabled.len() != 2 { return; }
        let e1 = &enabled[0];
        let e2 = &enabled[1];
        let neighbor1 = if e1.node1 == node_id { e1.node2 } else { e1.node1 };
        let neighbor2 = if e2.node1 == node_id { e2.node2 } else { e2.node1 };
        
        let new_edge_id = self.population_state.get_edge_id(&self.specialization, neighbor1, neighbor2);
        let new_edge = Edge::new(neighbor1, neighbor2, new_edge_id, None, None);
        
        let node = &self.nodes[&node_id];
        if self.specialization.remove_node(node, e1, e2, &new_edge, &self.nodes, &self.edges, rng) {
            for eid in connected { self.edges.remove(&eid); }
            self.nodes.remove(&node_id);
            self.edges.insert(new_edge_id, new_edge);
        }
    }
    
    fn perform_remove_node_single(&mut self, node_id: u64, rng: &mut impl Rng) {
        let connected: Vec<u64> = self.edges.values()
        .filter(|e| (e.node1 == node_id || e.node2 == node_id))
        .map(|e| e.id).collect();
        let enabled: Vec<Edge<S::EdgeData>> = connected.iter()
        .filter_map(|&id| self.edges.get(&id).cloned())
        .filter(|e| e.enabled)
        .collect();
        
        if enabled.len() != 1 { return; }
        let edge = &enabled[0];
        let node = &self.nodes[&node_id];
        if self.specialization.remove_node_single(node, edge, &self.nodes, &self.edges, rng) {
            for eid in connected { self.edges.remove(&eid); }
            self.nodes.remove(&node_id);
        }
    }
}

// --- Population ---

pub struct PopulationState<S: Specialization> {
    pub edge_id: Mutex<u64>,
    pub node_id: Mutex<u64>,
    pub edge_genes: Mutex<HashMap<(u64, u64), u64>>,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Specialization> PopulationState<S> {
    pub fn new() -> Self {
        Self {
            edge_id: Mutex::new(0),
            node_id: Mutex::new(0),
            edge_genes: Mutex::new(HashMap::new()),
            _marker: std::marker::PhantomData,
        }
    }
    pub fn new_node_id(&self) -> u64 {
        let mut g = self.node_id.lock().unwrap();
        *g += 1; *g
    }
    pub fn get_edge_id(&self, spec: &S, n1: u64, n2: u64) -> u64 {
        let key = spec.normalize_edge(n1, n2);
        let mut genes = self.edge_genes.lock().unwrap();
        if let Some(&id) = genes.get(&key) { id } else {
            let mut g = self.edge_id.lock().unwrap();
            *g += 1;
            genes.insert(key, *g);
            *g
        }
    }
}

#[derive(Clone)]
pub struct Species<S: Specialization> {
    pub representative: Genome<S>,
    pub members: Vec<GenomeRef<S>>,
    pub config: NEATConfig<S::Config>,
}

impl<S: Specialization> Species<S> {
    pub fn new(representative: Genome<S>, config: NEATConfig<S::Config>) -> Self {
        Species {
            representative: representative.clone(),
            members: Vec::new(),
            config,
        }
    }
}

pub struct Population<S: Specialization> {
    pub config: NEATConfig<S::Config>,
    pub specialization: S,
    pub state: Arc<PopulationState<S>>,
    pub species: Vec<Species<S>>,
    pub members: Vec<GenomeRef<S>>,
    pub current_species_threshold: f64,
}

impl<S: Specialization> Population<S> {
    pub fn new(config: NEATConfig<S::Config>, specialization: S) -> Self {
        let current_species_threshold = config.species_threshold;
        Population {
            config,
            specialization,
            state: Arc::new(PopulationState::new()),
            species: Vec::new(),
            members: Vec::new(),
            current_species_threshold,
        }
    }
    
    pub fn initialize(&mut self, start_genome: Option<&Genome<S>>) {
        let (nodes_map, edges_map) = if let Some(g) = start_genome {
            (g.nodes.clone(), g.edges.clone())
        } else {
            let (nodes_list, edges_list) = self.specialization.initialize_genome(self);
            let mut nodes = HashMap::new();
            let mut edges = HashMap::new();
            for n in nodes_list {
                nodes.insert(n.id, n);
            }
            for e in edges_list {
                edges.insert(e.id, e);
            }
            (nodes, edges)
        };

        let mut max_node = 0;
        let mut max_edge = 0;
        
        for n in nodes_map.values() {
            max_node = max_node.max(n.id);
        }
        for e in edges_map.values() {
            max_edge = max_edge.max(e.id);
            let key = self.specialization.normalize_edge(e.node1, e.node2);
            self.state.edge_genes.lock().unwrap().insert(key, e.id);
        }
        
        {
            let mut node_id = self.state.node_id.lock().unwrap();
            *node_id = (*node_id).max(max_node);
        }
        {
            let mut edge_id = self.state.edge_id.lock().unwrap();
            *edge_id = (*edge_id).max(max_edge);
        }
        
        let base_genome = Genome::new(
            self.config.clone(), self.specialization.clone(), self.state.clone(),
            Some(edges_map), Some(nodes_map)
        );
        
        let mut rng = rand::thread_rng();
        self.members.clear();
        for _ in 0..self.config.population_size {
            let mut new_genome = base_genome.clone();
            new_genome.mutate(&mut rng);
            self.members.push(Arc::new(RwLock::new(new_genome)));
        }
        
        self.respeciate();
    }
    
    /// Re-assigns all genomes to species based on compatibility.
    /// Optimized with Rayon for concurrency.
    pub fn respeciate(&mut self) {
        // 1. Snapshot representatives to allow concurrent access
        // We capture just the genome data needed for compatibility checks.
        let existing_reps: Vec<Genome<S>> = self.species.iter()
            .map(|s| s.representative.clone())
            .collect();
        
        // Clear old members
        for s in &mut self.species { s.members.clear(); }
        
        let threshold = self.current_species_threshold;

        // 2. Parallel Assignment to Existing Species
        // We calculate compatibility for all members against all existing reps in parallel.
        let (assignments, unassigned): (Vec<_>, Vec<_>) = self.members.par_iter()
            .map(|g_ref| {
                let g = g_ref.read().unwrap();
                // Find first compatible species index
                let match_idx = existing_reps.iter().position(|rep| {
                    calculate_compatibility(rep, &g) < threshold
                });
                (match_idx, g_ref.clone())
            })
            // Collect the calculated indices
            .collect::<Vec<_>>()
            .into_iter()
            .partition(|(idx, _)| idx.is_some());

        // Distribute assigned genomes to their buckets
        for (idx, g_ref) in assignments {
            if let Some(i) = idx {
                self.species[i].members.push(g_ref);
            }
        }
        
        // Extract raw GenomeRefs from the unassigned tuples
        let mut pool: Vec<GenomeRef<S>> = unassigned.into_iter().map(|(_, g)| g).collect();

        // 3. Parallel Sieve for New Species (Your Idea)
        // For remaining unassigned genomes, we must iteratively pick a rep and filter the pool.
        while let Some(new_rep_ref) = pool.first().cloned() {
            // Read representative data
            let rep_data = new_rep_ref.read().unwrap().clone();
            
            // Parallel Partition: Separates "children" of this candidate from "pool of others"
            let (members, remaining): (Vec<_>, Vec<_>) = pool.par_iter()
                .cloned()
                .partition(|g_ref| {
                    let g = g_ref.read().unwrap();
                    calculate_compatibility(&rep_data, &g) < threshold
                });
            
            // Create new species
            let mut new_s = Species::new(rep_data, self.config.clone());
            new_s.members = members; // Includes the rep itself as it matches distance 0
            self.species.push(new_s);
            
            pool = remaining;
        }
        
        // 4. Remove empty species
        self.species.retain(|s| !s.members.is_empty());
        
        // 5. Update representatives (random member becomes new rep)
        let mut rng = rand::thread_rng();
        for s in &mut self.species {
            if let Some(rep_ref) = s.members.choose(&mut rng) {
                s.representative = rep_ref.read().unwrap().clone();
            }
        }

        // 6. Dynamic Threshold Adjustment
        if let Some(target) = self.config.target_species {
            let num_species = self.species.len();
            let difference = target as isize - num_species as isize;
            let delta = difference as f64 / 100.0;
            self.current_species_threshold = (self.current_species_threshold - delta)
                .max(0.01)
                .min(100.0);
            // TODO: don't hard-code factors here; could use the largest
            //       distances seen above as proxy for example
        }
    }
    
    /// Main entry point for reproduction. Returns tuple (elites, species_elites, children).
    /// State mutations (like fitness processing) happen here.
    pub fn reproduce(&mut self, rng: &mut impl Rng) -> (Vec<GenomeRef<S>>, Vec<GenomeRef<S>>, Vec<GenomeRef<S>>) {
        match self.config.mating_strategy {
            MatingStrategy::NEATSpeciation => self.make_offspring_neat(rng),
            MatingStrategy::GlobalTournament => {
                let (elites, children) = self.make_offspring_global(rng);
                (elites, Vec::new(), children)
            },
        }
    }
    
    fn make_offspring_neat(&mut self, rng: &mut impl Rng) -> (Vec<GenomeRef<S>>, Vec<GenomeRef<S>>, Vec<GenomeRef<S>>) {
        let mut min_fitness = f64::INFINITY;
        let mut max_fitness = f64::NEG_INFINITY;
        
        let mut num_valid = 0;
        for g in self.members.iter() {
            let fit = g.read().unwrap().fitness;
            if fit > f64::NEG_INFINITY {
                min_fitness = min_fitness.min(fit);
                max_fitness = max_fitness.max(fit);
                num_valid += 1;
            }
        }
        
        if num_valid == 0 {
            min_fitness = 0.0;
            max_fitness = 0.0;
        }
            
        // Calculate dynamic baseline to ensure poorest performers don't immediately die.
        let range = (max_fitness - min_fitness).max(1e-7); // prevent divide-by-zero
        let baseline = range * 0.1; // worst individual gets roughly 10% the weight of a top-tier individual's baseline
        let offset = -min_fitness + baseline;

        // Adjust fitness by species size
        for s in &mut self.species {
            let n = s.members.len() as f64;
            for g_ref in &mut s.members {
                let mut g = g_ref.write().unwrap();
                if g.fitness == f64::NEG_INFINITY {
                    g.adj_fitness = Some(0.0);
                } else {
                    g.adj_fitness = Some((g.fitness + offset) / n);
                }
            }
            // Sort species members by fitness descending
            s.members.sort_by(|a, b| {
                let ga = a.read().unwrap();
                let gb = b.read().unwrap();
                gb.fitness.partial_cmp(&ga.fitness).unwrap_or(Ordering::Equal)
            });
        }
        
        let mut elites = Vec::new();
        let mut species_elites = Vec::new();
        let mut children = Vec::with_capacity(self.config.population_size);
        
        // Global Elitism
        let mut all_genomes: Vec<GenomeRef<S>> = self.members.iter().cloned().collect();
        all_genomes.sort_by(|a, b| {
            let ga = a.read().unwrap();
            let gb = b.read().unwrap();
            gb.fitness.partial_cmp(&ga.fitness).unwrap_or(Ordering::Equal)
        });
        
        for i in 0..self.config.num_elites_global.min(all_genomes.len()) {
            let elite_ref = all_genomes[i].clone();
            elite_ref.write().unwrap().is_elite = true;
            elites.push(elite_ref);
        }
        
        let total_adj_fitness: f64 = self.species.iter()
        .map(|s| s.members.iter().map(|g| g.read().unwrap().adj_fitness.unwrap_or(0.0)).sum::<f64>())
        .sum();
        
        // Calculate remaining slots
        let remaining_slots = self.config.population_size.saturating_sub(elites.len());
        
        for s in &self.species {
            let s_total = s.members.iter().map(|g| g.read().unwrap().adj_fitness.unwrap_or(0.0)).sum::<f64>();
            if total_adj_fitness <= 1e-9 { continue; }
            
            let num_offspring = ((s_total / total_adj_fitness) * remaining_slots as f64).round() as usize;
            if num_offspring == 0 { continue; }
            
            // Species Elitism
            let elite_count = self.config.num_elites_species.min(s.members.len()).min(num_offspring);
            for i in 0..elite_count {
                let elite_ref = s.members[i].clone();
                elite_ref.write().unwrap().is_elite = true;
                // Avoid duplicating if already in global elites? 
                // For simplicity we just add it, selection step handles sizing.
                species_elites.push(elite_ref);
            }
            
            let breed_count = num_offspring.saturating_sub(elite_count);
            let pool_size = (s.members.len() as f64 * self.config.selection_share).ceil() as usize;
            let pool = &s.members[0..pool_size.min(s.members.len())];
            
            for _ in 0..breed_count {
                if pool.is_empty() { break; }
                let p1_ref = pool.choose(rng).unwrap();
                let p2_ref = pool.choose(rng).unwrap();
                let p1 = p1_ref.read().unwrap();
                let p2 = p2_ref.read().unwrap();
                let mut child = crossover(&*p1, &*p2, rng);
                child.is_elite = false;
                children.push(Arc::new(RwLock::new(child)));
            }
        }
        
        // Fill remainder if rounding errors left us short
        while elites.len() + species_elites.len() + children.len() < self.config.population_size {
            if let Some(s) = self.species.choose(rng) {
                if !s.members.is_empty() {
                    let p1_ref = s.members.choose(rng).unwrap();
                    let p2_ref = s.members.choose(rng).unwrap();
                    let p1 = p1_ref.read().unwrap();
                    let p2 = p2_ref.read().unwrap();
                    let mut child = crossover(&*p1, &*p2, rng);
                    child.is_elite = false;
                    children.push(Arc::new(RwLock::new(child)));
                } else { break; }
            } else { break; }
        }
        
        (elites, species_elites, children)
    }
    
    fn make_offspring_global(&mut self, rng: &mut impl Rng) -> (Vec<GenomeRef<S>>, Vec<GenomeRef<S>>) {
        let mut elites = Vec::new();
        let mut children = Vec::new();
        
        self.members.sort_by(|a, b| {
            let ga = a.read().unwrap();
            let gb = b.read().unwrap();
            gb.fitness.partial_cmp(&ga.fitness).unwrap_or(Ordering::Equal)
        });
        
        for i in 0..self.config.num_elites_global.min(self.members.len()) {
            let elite_ref = self.members[i].clone();
            elite_ref.write().unwrap().is_elite = true;
            elites.push(elite_ref);
        }
        
        while elites.len() + children.len() < self.config.population_size {
            let p1_ref = self.tournament_select(rng);
            let p2_ref = self.tournament_select(rng);
            let p1 = p1_ref.read().unwrap();
            let p2 = p2_ref.read().unwrap();
            let mut child = crossover(&*p1, &*p2, rng);
            child.is_elite = false;
            children.push(Arc::new(RwLock::new(child)));
        }
        
        (elites, children)
    }
    
    fn tournament_select(&self, rng: &mut impl Rng) -> &GenomeRef<S> {
        let k = self.config.tournament_size;
        let mut best = self.members.choose(rng).unwrap();
        for _ in 1..k {
            let next = self.members.choose(rng).unwrap();
            if next.read().unwrap().fitness > best.read().unwrap().fitness { best = next; }
        }
        best
    }
    
    /// Selects genomes to form the new population
    pub fn select(&mut self, elites: Vec<GenomeRef<S>>, species_elites: Vec<GenomeRef<S>>, mut children: Vec<GenomeRef<S>>, _rng: &mut impl Rng) {
        match self.config.mating_strategy {
            MatingStrategy::NEATSpeciation => {
                // Combine pools
                children.extend(elites);
                children.extend(species_elites);
                let mut pool = children;

                // NEAT standard selection is implicit in reproduction counts,
                // but we must ensure we don't exceed pop size
                pool.truncate(self.config.population_size);
                self.members = pool;

                // ReSpeciate
                self.respeciate();
            },
            MatingStrategy::GlobalTournament => {
                let mut pool = self.members.clone();
                pool.extend(children);
                pool.extend(elites);
                pool.extend(species_elites);
                
                pool.sort_by(|a, b| {
                    let ga = a.read().unwrap();
                    let gb = b.read().unwrap();
                    gb.fitness.partial_cmp(&ga.fitness).unwrap_or(Ordering::Equal)
                });
                
                self.members = pool.into_iter().take(self.config.population_size).collect();
                self.species.clear();
            }
        }
    }
    
}

// --- Algorithm Core Functions ---

pub fn calculate_compatibility<S: Specialization>(g1: &Genome<S>, g2: &Genome<S>) -> f64 {
    let e1 = g1.sorted_edges();
    let e2 = g2.sorted_edges();
    let mut matching = Vec::new();
    let mut disjoint = 0;
    let mut excess = 0;
    let max_id1 = e1.last().map(|e| e.id).unwrap_or(0);
    let max_id2 = e2.last().map(|e| e.id).unwrap_or(0);
    let mut i1 = 0;
    let mut i2 = 0;
    
    while i1 < e1.len() || i2 < e2.len() {
        let id1 = if i1 < e1.len() { Some(e1[i1].id) } else { None };
        let id2 = if i2 < e2.len() { Some(e2[i2].id) } else { None };
        match (id1, id2) {
            (Some(u), Some(v)) => match u.cmp(&v) {
                Ordering::Equal => { matching.push(u); i1+=1; i2+=1; },
                Ordering::Less => { if u > max_id2 { excess += 1; } else { disjoint += 1; } i1+=1; },
                Ordering::Greater => { if v > max_id1 { excess += 1; } else { disjoint += 1; } i2+=1; }
            },
            (Some(_), None) => { excess += 1; i1+=1; },
            (None, Some(_)) => { excess += 1; i2+=1; },
            (None, None) => break,
        }
    }
    let n = std::cmp::max(e1.len(), e2.len()).max(1) as f64;
    let base = (g1.config.c1 * excess as f64 / n) + (g1.config.c2 * disjoint as f64 / n);
    // Always use custom compatibility since we eliminated use_custom_compat parameter
    let custom = g1.specialization.compatibility(g1, g2, &matching);
    base + custom
}

pub fn crossover<S: Specialization>(g1: &Genome<S>, g2: &Genome<S>, rng: &mut impl Rng) -> Genome<S> {
    let (p1, p2) = if g1.fitness > g2.fitness { (g1, g2) } else { (g2, g1) };
    
    // If crossover is disabled, just clone the fitter parent
    if !p1.config.crossover_enabled {
        return p1.clone();
    }
    
    let mut child_edges = HashMap::new();
    let p1_edges = p1.sorted_edges();
    let p2_map: HashMap<u64, &Edge<S::EdgeData>> = p2.edges.iter().map(|(&k, v)| (k, v)).collect();
    for e1 in p1_edges {
        if let Some(e2) = p2_map.get(&e1.id) {
            let chosen = if rng.gen_bool(0.5) { e1 } else { e2 };
            let mut new_e = (*chosen).clone();
            // if !e1.enabled || !e2.enabled { if rng.gen_bool(0.75) { new_e.enabled = false; } }
            new_e.enabled = e1.enabled;
            child_edges.insert(new_e.id, new_e);
        } else {
            child_edges.insert(e1.id, (*e1).clone());
        }
    }
    let child_nodes = p1.nodes.clone().into_iter().collect();
    Genome::new(p1.config.clone(), p1.specialization.clone(), p1.population_state.clone(), Some(child_edges), Some(child_nodes))
}

