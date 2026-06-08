from __future__ import annotations

try:
    import neatwork.neatwork_lib as neatwork_lib
except ImportError:
    import neatwork_lib

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, TYPE_CHECKING

# optuna is only needed for hyperparameter tuning (the optional `trial` arg of
# `run`). It is a heavy, HPO-only dependency, so it is imported lazily inside
# `run` rather than at module load — a plain client (e.g. MMGIS) never needs it.
if TYPE_CHECKING:
    import optuna

from neatwork.types import *

Node = neatwork_lib.Node
Edge = neatwork_lib.Edge
Genome = neatwork_lib.Genome
Species = neatwork_lib.Species


class Population:
    def __init__(self, config: Union[NEATCoreConfig, dict], start_genome: Optional[Genome] = None):
        if isinstance(config, NEATCoreConfig):
            config = config.model_dump()
        self._inner = neatwork_lib.Population(config, start_genome)

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


def _eval_args(eval_config: Optional[EvalConfig]):
    """Build the objectives/scalarization/constraints kwargs for evaluate_genomes."""
    if eval_config is None:
        return {}
    result = {"constraints": [c.value for c in eval_config.constraints]}
    if eval_config.objectives:
        result["objectives"] = [o.value for o in eval_config.objectives]
        result["scalarization"] = eval_config.scalarization
    return result


def evaluate_genomes(genomes, cost_map, eval_config: Optional[EvalConfig] = None):
    neatwork_lib.evaluate_genomes(genomes, cost_map, **_eval_args(eval_config))

mutate_genomes = lambda genomes, pop: neatwork_lib.mutate_genomes(genomes, pop._inner)
make_offspring = lambda pop, count: neatwork_lib.make_offspring(pop._inner, count)
select = lambda pop, elites, species_elites, children: neatwork_lib.select(pop._inner, elites, species_elites, children)


def genome_to_network(genome: Genome) -> Network:
    nodes = [
        NetworkNode(id=nid, pos=tuple(n.data.pos), type=NodeType(n.data.type_))
        for nid, n in genome.nodes.items()
    ]
    edges = [
        NetworkEdge(id=e.id, node1=e.node1, node2=e.node2, enabled=e.enabled)
        for e in genome.edges.values()
    ]
    return Network(nodes=nodes, edges=edges, fitness=genome.fitness)


def network_to_genome(network: Network, pop: Population) -> Genome:
    return pop.create_genome(network.to_dict())


def init_genome(
        fixed_nodes: List[Tuple[int, int]],
        grid_size: Tuple[int, int],
        graph_init=GraphInit.FCN,
    ) -> Genome:
    pop = Population(NEATCoreConfig(
        special=NWConfig(
            grid_size=grid_size,
            fixed_nodes=fixed_nodes,
            graph_init=graph_init,
        )
    ))
    return pop.initialize_genome()


def get_best_from_species(pop: Population, n_species: Optional[int] = None) -> List[Genome]:
    get_best = lambda gs: max(gs, key=lambda g: g.fitness)
    genomes = sorted([
        get_best(s.members)
        for s in pop.species
    ], key=lambda g: g.fitness)[:n_species if n_species is not None else len(pop.species)]
    return genomes


def _precompute_threshold(
    config: NEATCoreConfig,
    eval_config: EvalConfig,
    target_species: int,
    gens: int = 15,
    trials: int = 10,
    print_: bool = True,
) -> float:
    if print_:
        print(f"--- Precomputing Threshold for {target_species} Species ---")

    low, high = 0.1, 100.0
    current = max(low, min(high, config.species_threshold))
    best_thresh = current
    min_diff = float('inf')

    for i in range(trials):
        temp_cfg = config.model_dump()
        temp_cfg['species_threshold'] = current

        pop = Population(temp_cfg)
        curr_g = pop.get_initial_genomes()
        evaluate_genomes(curr_g, eval_config.cost_map, eval_config)

        final_species = 0
        for _ in range(gens):
            elites, species_elites, children = make_offspring(pop, temp_cfg['population_size'])
            mutate_genomes(children, pop)
            offspring = elites + species_elites + children
            evaluate_genomes(offspring, eval_config.cost_map, eval_config)
            select(pop, elites, species_elites, children)
            _, _, final_species, _ = pop.get_stats()

        if print_:
            print(f"  [Try {i+1}] Threshold {current:.4f} => {final_species} species (Goal: {target_species})")

        diff = abs(final_species - target_species)
        if diff < min_diff:
            min_diff = diff
            best_thresh = current

        if final_species == target_species:
            best_thresh = current
            break

        if final_species > target_species:
            low = current
        else:
            high = current

        current = (low + high) / 2.0

    if print_:
        print(f"--- Settled on Threshold {best_thresh:.4f} (Diff: {min_diff}) ---")
    return best_thresh


get_genome_mask = neatwork_lib.get_genome_mask
get_trace_difference = neatwork_lib.get_trace_difference
genome_compatibility = neatwork_lib.genome_compatibility
compute_objective = neatwork_lib.compute_objective


@dataclass
class GenerationStatistics:
    generation: int
    time_sec: float
    max_fitness: float
    avg_fitness: float
    species_count: int

@dataclass
class RunStatistics:
    config: NEATCoreConfig
    start_time_sec: float
    gens: List[GenerationStatistics] = field(default_factory=list)

    def record_gen(self, gen: int, pop, print_: bool = True):
        max_fit, avg_fit, n_species, species_data = pop.get_stats()

        self.gens.append(GenerationStatistics(
            generation=gen,
            max_fitness=max_fit,
            avg_fitness=avg_fit,
            time_sec=time.time() - self.start_time_sec,
            species_count=n_species,
        ))

        if print_:
            performers_str = ""
            if n_species > 0:
                top_s_fits = sorted(species_data, reverse=True)[:5]
                performers_str = " | top species: " + ', '.join(f'{fit:08.4f}' for fit in top_s_fits)

            print(f"Gen {str(gen+1).zfill(4)} | "
                  f"Avg fit {avg_fit:08.4f} | "
                  f"Top fit {max_fit:08.4f} | "
                  f"Species {n_species}" + performers_str)

    def record_final(self, top_genome, print_: bool = True):
        if print_:
            print("\n--- Evolution Finished ---")
            if top_genome:
                print(f"Top genome fitness: {top_genome.fitness}")

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
    trial: Optional[optuna.Trial] = None,
    trial_step_offset: Optional[int] = None,
) -> RunResults:
    ngen = config.ngen if config.ngen is not None else int(1e6)
    target_fit = config.target_fit
    cost_map = eval_config.cost_map

    start_genome = None
    if config.start_genome is not None:
        temp_pop = Population(config)
        start_genome = network_to_genome(config.start_genome, temp_pop)

    if config.precompute_threshold:
        new_thresh = _precompute_threshold(config, eval_config, config.target_species, print_=print_)
        config = config.model_copy(update={"species_threshold": new_thresh})

    pop = Population(config, start_genome)
    current_genomes = pop.get_initial_genomes()

    stats = RunStatistics(config, start_time_sec=time.time())
    top_genomes_hist = []

    evaluate_genomes(current_genomes, cost_map, eval_config)

    best_g = max(current_genomes, key=lambda g: g.fitness)
    top_genomes_hist.append(best_g)

    for gen in range(ngen):
        stats.record_gen(gen, pop, print_)

        best_g = max(current_genomes, key=lambda g: g.fitness)
        
        if trial is not None:
            import optuna  # lazy: only required for HPO trials
            trial.report(best_g.fitness, step=gen+int(trial_step_offset))
            if trial.should_prune():
                raise optuna.TrialPruned()

        if target_fit is not None and best_g.fitness >= target_fit:
            if print_:
                print(f"\nTarget fitness {target_fit} reached.")
            break

        elites, species_elites, children = make_offspring(pop, config.population_size)
        mutate_genomes(children, pop)

        offspring = elites + species_elites + children
        evaluate_genomes(offspring, cost_map, eval_config)

        select(pop, elites, species_elites, children)

        current_genomes = pop.members
        best_g = max(current_genomes, key=lambda g: g.fitness)
        top_genomes_hist.append(best_g)

    best_g = max(current_genomes, key=lambda g: g.fitness)
    stats.record_final(best_g, print_=print_)

    return RunResults(
        top_networks=[genome_to_network(g) for g in top_genomes_hist],
        pop=pop,
        stats=stats,
        best_from_species=[genome_to_network(g) for g in get_best_from_species(pop)],
    )
