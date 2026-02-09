try:
    # in case neat_tnp is used as package
    import neat_tnp.neat_rs_tnp_rs as neat_rs_tnp_rs
except ImportError:
    # in case neat_tnp source code is used directly
    import neat_rs_tnp_rs
import numpy as np
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Union, Any
from enum import Enum
import json

# --- Enums & Configs (Python-side) ---

class SelectionStrategy(str, Enum):
    NEAT = "NEAT"
    NSGA2 = "NSGA2"
    SPEA2 = "SPEA2"

class MatingStrategy(str, Enum):
    NEATSpeciation = "NEATSpeciation"
    GlobalTournament = "GlobalTournament"

class NodeType(str, Enum):
    Fixed = "Fixed"
    Flexible = "Flexible"

class EvaluationStrategy(str, Enum):
    SingleObjective = "SingleObjective"
    MultiObjectiveBridge = "MultiObjectiveBridge"

class GraphInit:
    FCN = "FCN"
    @staticmethod
    def grid(rows: int, cols: int):
        return {"Grid": (rows, cols)}

@dataclass
class TNPConfig:
    grid_size: Tuple[int, int] = (10, 10)
    fixed_nodes: List[Tuple[int, int]] = field(default_factory=list)
    move_node_mutation_prob: float = 0.6
    move_node_mutation_sigma: float = 5.0
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.SingleObjective
    graph_init: Any = GraphInit.FCN
    compatibility_sigma: int = 5
    c3: float = 1.0

@dataclass
class NEATConfig:
    population_size: int = 100
    add_node_mutation_prob: float = 0.4
    add_edge_mutation_prob: float = 0.4
    remove_edge_mutation_prob: float = 0.1
    remove_node_mutation_prob: float = 0.1
    remove_node_single_mutation_prob: float = 0.1
    num_elites_species: int = 1
    num_elites_global: int = 1
    selection_share: float = 0.2
    tournament_size: int = 2
    species_threshold: float = 3.0
    target_species: Optional[int] = None
    mating_strategy: MatingStrategy = MatingStrategy.NEATSpeciation
    selection_strategy: SelectionStrategy = SelectionStrategy.NEAT
    diversity_objective: bool = False
    crossover_enabled: bool = True
    c1: float = 1.0
    c2: float = 1.0
    special: TNPConfig = field(default_factory=TNPConfig)

# for json serializbility
class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Enums
        if isinstance(obj, Enum):
            return obj.value
        # Handle dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return super().default(obj)

# Re-expose Rust types
Node = neat_rs_tnp_rs.Node
Edge = neat_rs_tnp_rs.Edge
Genome = neat_rs_tnp_rs.Genome
Species = neat_rs_tnp_rs.Species

class Population:
    def __init__(self, config: Union[NEATConfig, dict], start_genome: Optional[Genome] = None):
        if not isinstance(config, dict):
            config = asdict(config)
        self._inner = neat_rs_tnp_rs.Population(config, start_genome)

    def get_initial_genomes(self) -> List[Genome]:
        return self._inner.get_initial_genomes()

    def get_stats(self) -> Tuple[float, float, int, List[Tuple[float, float, int]]]:
        return self._inner.get_stats()

    def new_node_id(self) -> int:
        return self._inner.new_node_id()

    def get_edge_id(self, n1: int, n2: int) -> int:
        return self._inner.get_edge_id(n1, n2)

    def create_genome(self, structure: dict) -> Genome:
        return self._inner.create_genome(structure)

    def initialize_genome(self) -> Genome:
        return self._inner.initialize_genome()

    @property
    def members(self) -> List[Genome]:
        return self._inner.members

    @property
    def species(self) -> List[Species]:
        return self._inner.species
    
    @property
    def current_species_threshold(self) -> float:
        return self._inner.current_species_threshold

# Wrapped functions to handle Population wrapper
def evaluate_genomes(genomes: List[Genome], cost_map: np.ndarray):
    neat_rs_tnp_rs.evaluate_genomes(genomes, cost_map)

def mutate_genomes(genomes: List[Genome], pop: Population):
    neat_rs_tnp_rs.mutate_genomes(genomes, pop._inner)

def make_offspring(pop: Population, count: int) -> Tuple[List[Genome], List[Genome], List[Genome]]:
    return neat_rs_tnp_rs.make_offspring(pop._inner, count)

def select(pop: Population, elites: List[Genome], species_elites: List[Genome], children: List[Genome]):
    neat_rs_tnp_rs.select(pop._inner, elites, species_elites, children)

def augment_diversity(pop: Population, genomes: List[Genome]):
    pop._inner.augment_diversity(genomes)

def process_multi_objective(pop: Population, genomes: List[Genome]):
    neat_rs_tnp_rs.process_multi_objective(pop._inner, genomes)

# helpers

def init_genome(
        fixed_nodes: List[Tuple[int, int]],
        grid_size: Tuple[int, int],
        graph_init: GraphInit,
    ) -> Genome:
    pop = Population(NEATConfig(
        special=TNPConfig(
            grid_size=grid_size,
            fixed_nodes=fixed_nodes,
            graph_init=graph_init,
        )
    ))
    g = pop.initialize_genome()
    return g

def evaluate_single(
        genome: Genome,
    ):
    return Population(NEATConfig(
        special=TNPConfig(
            grid_size=grid_size,
        )
    ))

def get_best_from_species(pop: Population, n_species: Optional[int] = None) -> List[Genome]:
    get_best = lambda gs: max(gs, key=lambda g: g.fitness)
    genomes = sorted([
        get_best(s.members)
        for s in pop.species
        # if best_performer(species.members).fitness >= 20.0
    ], key = lambda g: g.fitness)[:n_species if n_species is not None else len(pop.species)]
    return genomes

def _precompute_threshold(
    config: NEATConfig, 
    cost_map: np.ndarray, 
    target_species: int,
    gens: int = 15,
    trials: int = 10
) -> float:
    """
    Performs short runs to find a species threshold that yields roughly `target_species`.
    Returns the calibrated threshold.
    """
    print(f"--- Precomputing Threshold for {target_species} Species ---")
    
    # Heuristic bounds for threshold
    low = 0.1
    high = 100.0
    
    current = config.species_threshold
    # clamp start
    current = max(low, min(high, current))
    
    best_thresh = current
    min_diff = float('inf')

    for i in range(trials):
        # Create a temp config dict for this trial
        temp_cfg = asdict(config)
        temp_cfg['species_threshold'] = current
        
        # Init population with temporary config
        pop = Population(temp_cfg)
        curr_g = pop.get_initial_genomes()
        evaluate_genomes(curr_g, cost_map)
        augment_diversity(pop, curr_g)
        process_multi_objective(pop, curr_g)
        
        final_species = 0
        
        # Evolve for 'gens' to let speciation stabilize
        for _ in range(gens):
            elites, species_elites, children = make_offspring(pop, temp_cfg['population_size'])
            mutate_genomes(children, pop)
            offspring = elites + species_elites + children
            evaluate_genomes(offspring, cost_map)
            augment_diversity(pop, offspring)
            # Joint MO calc
            process_multi_objective(pop, pop.members + offspring)
            select(pop, elites, species_elites, children)
            
            _, _, final_species, _ = pop.get_stats()
            
        print(f"  [Try {i+1}] Threshold {current:.4f} => {final_species} species (Goal: {target_species})")
        
        # Track best
        diff = abs(final_species - target_species)
        if diff < min_diff:
            min_diff = diff
            best_thresh = current
        
        if final_species == target_species:
            best_thresh = current
            break
        
        # Binary search / adjustment step
        # Threshold UP -> Species DOWN (Monotonically decreasing assumption)
        
        if final_species > target_species:
            # Too many species. Need fewer. Need HIGHER threshold (looser matching).
            # The solution is in [current, high]
            low = current
        else:
            # Too few species. Need more. Need LOWER threshold (stricter matching).
            # The solution is in [low, current]
            high = current

        current = (low + high) / 2.0

    print(f"--- Settled on Threshold {best_thresh:.4f} (Diff: {min_diff}) ---")
    return best_thresh

get_genome_mask = neat_rs_tnp_rs.get_genome_mask
get_trace_difference = neat_rs_tnp_rs.get_trace_difference
genome_compatibility = neat_rs_tnp_rs.genome_compatibility
get_pareto_front = neat_rs_tnp_rs.get_pareto_front

@dataclass
class GenerationStatistics:
    generation: int
    time_sec: float
    max_fitness: float
    avg_fitness: float
    species_count: int

@dataclass
class RunStatistics:
    config: NEATConfig
    start_time_sec: float
    gens: List[GenerationStatistics] = field(default_factory=list)

    def record_gen(self, gen: int, pop: neat_rs_tnp_rs.Population, print_: bool = True):
        # Stats are pulled from the Rust population
        max_fit, avg_fit, n_species, species_data = pop.get_stats()
        
        gen_stats = GenerationStatistics(
            generation=gen,
            max_fitness=max_fit,
            avg_fitness=avg_fit,
            time_sec=time.time() - self.start_time_sec,
            species_count=n_species
        )
        self.gens.append(gen_stats)

        if print_:
            performers_str = ""
            if n_species > 0:
                top_s_fits = sorted(species_data, reverse=True)[:5]
                performers_str = " | top species: " + ', '.join(f'{fit:08.4f}' for fit in top_s_fits)

            print(f"Gen {str(gen+1).zfill(4)} | "+
                  f"Avg fit {avg_fit:08.4f} | "+
                  f"Top fit {max_fit:08.4f} | "+
                  f"Species {n_species}" + performers_str)
    
    def record_final(self, top_genome, print_: bool = True):
        if print_:
            print("\n--- Evolution Finished ---")
            if top_genome:
                print(f"Top genome fitness: {top_genome.fitness}")

@dataclass
class RunResults:
    top_genomes: List[Genome]
    pop: Population
    stats: RunStatistics
    best_from_species: List[Genome]

# --- Main Run Function ---

def run(
    config: NEATConfig, 
    cost_map: np.ndarray, 
    ngen: Optional[int] = None, 
    target_fit: Optional[float] = None, 
    start_genome: Optional[Genome] = None,
    precompute_threshold: bool = False,
    print_: bool = True,
) -> RunResults:
    """
    Runs the NEAT algorithm using the high-performance Rust core.
    
    Args:
        config: NEAT configuration
        cost_map: Cost map for evaluation (numpy array)
        ngen: Number of generations (default: 1e6)
        target_fit: Target fitness to stop early (optional)
        start_genome: Initial genome (optional)
        precompute_threshold: initialize species threshold by pre-simulation (optional)
        print_: Whether to print progress
        
    Returns:
        (top_genomes, population, stats)
    """
    if ngen is None: 
        ngen = int(1e6)
        
    # Precompute threshold if requested
    if precompute_threshold:
        new_thresh = _precompute_threshold(config, cost_map, config.target_species)
        config.species_threshold = new_thresh
    
    # Population wrapper handles config conversion
    pop = Population(config, start_genome)
    current_genomes = pop.get_initial_genomes()
    
    stats = RunStatistics(config, start_time_sec=time.time())
    top_genomes_hist = []
    
    evaluate_genomes(current_genomes, cost_map)
    augment_diversity(pop, current_genomes)
    # Explicitly process multi-objective fitness for the initial population.
    process_multi_objective(pop, current_genomes)
    
    best_g = max(current_genomes, key=lambda g: g.fitness)
    top_genomes_hist.append(best_g)
    
    for gen in range(ngen):
        stats.record_gen(gen, pop, print_)
        
        # Check target
        best_g = max(current_genomes, key=lambda g: g.fitness)
        if target_fit is not None and best_g.fitness >= target_fit:
            if print_: 
                print(f"\nTarget fitness {target_fit} reached.")
            break
        
        # Reproduce
        elites, species_elites, children = make_offspring(pop, config.population_size)
        
        # Mutate children only
        mutate_genomes(children, pop)
        
        # Evaluate all offspring
        offspring = elites + species_elites + children
        evaluate_genomes(offspring, cost_map)
        augment_diversity(pop, offspring)
        
        # Explicitly process multi-objective fitness on the combined pool (parents + offspring)
        # This ensures fitnesses are comparable before selection.
        all_candidates = pop.members + offspring
        process_multi_objective(pop, all_candidates)
        
        # Select for next generation
        select(pop, elites, species_elites, children)
        
        current_genomes = pop.members
        best_g = max(current_genomes, key=lambda g: g.fitness)
        top_genomes_hist.append(best_g)

    best_g = max(current_genomes, key=lambda g: g.fitness)
    stats.record_final(best_g, print_=print_)

    best_from_species = get_best_from_species(pop)
    
    return RunResults(
        top_genomes=top_genomes_hist,
        pop=pop,
        stats=stats,
        best_from_species=best_from_species
    )
