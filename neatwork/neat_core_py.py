from random import choice, random
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Callable, Any, Union
import time
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

from neatwork.types import MatingStrategy, NEATCoreConfig

# --- specialization interface ---

class NEATSpecialization(ABC):
    """
    Abstract class for specializing the NEAT algorithm.

    This allows customizing mutations, genome initialization, and graph
    behaviors (e.g., directed vs. undirected) for a specific use case.
    An instance of a subclass will be shared by all genomes.
    """

    def __init__(self, config: NEATCoreConfig):
        self.config = config
    
    # --- hooks ---

    @abstractmethod
    def initialize_genome(self, pop: 'Population') -> Tuple[List['Node'], List['Edge']]:
        """
        Define the structure of the default genome.
        
        Returns a graph as an adjacency list, where each edge has a unique ID.
        """
        pass
    
    @abstractmethod
    def mutate(self,
               nodes: Dict['NodeID', 'Node'],
               edges: Dict['EdgeID', 'Edge'],
               ):
        """Apply custom mutations to the genome."""
        pass

    @abstractmethod
    def add_node(self, 
                 node: 'Node',
                 old_edge: 'Edge',
                 edge_node1_new: 'Edge',
                 edge_new_node2: 'Edge',
                 nodes: Dict['NodeID', 'Node'],
                 edges: Dict['EdgeID', 'Edge'],
                 ) -> bool:
        """
        Called when a node is added by splitting an edge.
        Return True to allow the change, False to veto.
        """
        pass
    
    @abstractmethod
    def remove_node(self, 
                    node: 'Node', 
                    edge1: 'Edge',
                    edge2: 'Edge',
                    new_edge: 'Edge',
                    nodes: Dict['NodeID', 'Node'],
                    edges: Dict['EdgeID', 'Edge'],
                    ) -> bool:
        """
        Called when a node is about to be removed.
        Return True to allow the change, False to veto.
        """
        pass

    @abstractmethod
    def remove_node_single(self,
                           node: 'Node',
                           edge: 'Edge',
                           nodes: Dict['NodeID', 'Node'],
                           edges: Dict['EdgeID', 'Edge'],
                           ) -> bool:
        """
        Called when a degree-1 node is about to be removed.
        Return True to allow the change, False to veto.
        """
        pass
    
    @abstractmethod
    def add_edge(self,
                 edge: 'Edge',
                 nodes: Dict['NodeID', 'Node'],
                 edges: Dict['EdgeID', 'Edge'],
                 ) -> bool:
        """
        Called when a new edge is added.
        Return True to allow the change, False to veto.
        """
        pass
    
    @abstractmethod
    def remove_edge(self, 
                    edge: 'Edge', 
                    nodes: Dict['NodeID', 'Node'], 
                    edges: Dict['EdgeID', 'Edge'],
                    ) -> bool:
        """
        Called when an edge is about to be removed (disabled).
        Return True to allow the change, False to veto.
        """
        pass

    # allows for switching between undirected and directed graphs
    # default: directed graph
    @staticmethod
    def normalize_edge(node_id1: 'NodeID', node_id2: 'NodeID') -> Tuple['NodeID', 'NodeID']:
        """Normalizes an edge representation for innovation tracking."""
        return (node_id1, node_id2)
    
    @abstractmethod
    def compatibility(self,
                      genome1: 'Genome', matching1: List['Edge'], disjoint1: List['Edge'], excess1: List['Edge'],
                      genome2: 'Genome', matching2: List['Edge'], disjoint2: List['Edge'], excess2: List['Edge']):
        """
        Called when the compatibility distance (delta) between two
        genomes is calculated. The specialization can add additional
        terms to delta, apart from the already considered shares of
        disjoint and excess genes.
        """
        pass

# --- gene classes ---

@dataclass(kw_only=True)
class Gene:
    # a field for algorithm specializations to attach custom properties
    data: Any = None  # Optional[dataclass]

NodeID = int
EdgeID = int

@dataclass
class Node(Gene):
    """Represents a node gene."""
    id : NodeID

    def copy(self) -> 'Node':
        return Node(
            id=self.id,
            data=deepcopy(self.data),
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Node) and self.id == other.id

@dataclass
class Edge(Gene):
    """
    Represents an edge gene, linking two nodes.
    
    The `id` is the historical innovation number.
    """
    node1: NodeID
    node2: NodeID
    id: int
    enabled: bool = True

    def copy(self) -> 'Edge':
        """Creates a deep copy of this edge."""
        return Edge(
            node1=self.node1,
            node2=self.node2,
            id=self.id,
            enabled=self.enabled,
            data=deepcopy(self.data),
        )

# --- state containers ---

class Genome:
    """
    Represents a single individual (a graph) in the population.

    A genome consists of a set of node and edge genes and holds its
    evaluated fitness.
    """
    def __init__(
            self, 
            config: NEATCoreConfig,
            specialization: NEATSpecialization,
            population: 'Population',
            edges: Optional[Dict[EdgeID, Edge]] = None,
            nodes: Optional[Dict[NodeID, Node]] = None,
        ):
        # edge & nodes must be given both or none
        assert (edges is None) == (nodes is None)
        default_init = edges is None

        self.config: NEATCoreConfig = config
        self.specialization: NEATSpecialization = specialization
        self.population: 'Population' = population  # for accessing global innovation numbers
        self.fitness: float = 0.0
        
        self.nodes: Dict[NodeID, Node] = nodes or {}
        self.edges: Dict[EdgeID, Edge] = edges or {}

        if default_init:
            # create the default genome structure from the specialization
            nodes_, edges_ = self.specialization.initialize_genome(pop=self.population)
            self.nodes = {node.id: node for node in nodes_}
            self.edges = {edge.id: edge for edge in edges_}

    def copy(self) -> 'Genome':
        """Creates and returns a deep copy of this genome."""
        copied_genome = Genome(
            config=self.config,
            specialization=self.specialization,
            population=self.population, 
            edges={
                edge_id: edge.copy() 
                for edge_id, edge in self.edges.items()
            }, 
            nodes={
                node_id: node.copy() 
                for node_id, node in self.nodes.items()
            }, 
        )
        copied_genome.fitness = self.fitness
        return copied_genome

    @property
    def sorted_edges(self) -> List[Edge]:
        """Returns edges sorted by innovation ID."""
        return sorted(self.edges.values(), key=lambda c: c.id)

    @property
    def sorted_nodes(self) -> List[Node]:
        """Returns nodes sorted by node ID."""
        return sorted(self.nodes.values(), key=lambda n: n.id)

    # --- mutations (graph operations) ---

    def add_node(self, edge_id: EdgeID):
        """Splits an existing edge, adding a new node in its place."""
        old_edge = self.edges[edge_id]

        node1 = self.nodes[old_edge.node1]
        node2 = self.nodes[old_edge.node2]

        # create new node
        new_node_id = self.population.new_node_id()
        new_node = Node(id=new_node_id)

        # connect node1 -> new
        edge1_id = self.population.get_edge_id(node1.id, new_node.id)
        edge1 = Edge(node1=node1.id, node2=new_node.id, id=edge1_id)

        # connect new -> node2
        edge2_id = self.population.get_edge_id(new_node.id, node2.id)
        edge2 = Edge(node1=new_node.id, node2=node2.id, id=edge2_id)
        
        # only commit change if specialization approves
        if self.specialization.add_node(old_edge=old_edge,
                                        node=new_node,
                                        edge_node1_new=edge1,
                                        edge_new_node2=edge2,
                                        nodes=self.nodes,
                                        edges=self.edges,
                                        ):
            self.nodes[new_node.id] = new_node
            self.edges[edge1.id] = edge1
            self.edges[edge2.id] = edge2
            old_edge.enabled = False

    def remove_node(self, node_id: NodeID):
        """Removes a degree-2 node by rewiring its neighbors."""
        self.perform_remove_node(node_id)

    def remove_node_single(self, node_id: NodeID):
        """Removes a degree-1 node and its incident edge."""
        self.perform_remove_node_single(node_id)

    def add_edge(self, node1_id: NodeID, node2_id: NodeID) -> bool:
        """Adds a new edge between two existing nodes."""
        edge_id = self.population.get_edge_id(node1_id, node2_id)
        
        if edge_id in self.edges:
            return False  # edge already exists
        
        edge = Edge(node1=node1_id, node2=node2_id, id=edge_id)

        if self.specialization.add_edge(edge=edge, nodes=self.nodes, edges=self.edges):
            self.edges[edge.id] = edge
            return True
        else:
            return False  # specialization vetoed the edge

    def remove_edge(self, edge_id: EdgeID):
        """Disables a specific edge."""
        edge = self.edges[edge_id]
        if self.specialization.remove_edge(edge=edge, nodes=self.nodes, edges=self.edges):
            edge.enabled = False

    def add_node_mutation(self):
        """Applies the 'add node' mutation."""
        if random() < self.config.add_node_mutation_prob:
            # find enabled edges eligible for splitting
            enabled_edges = [c for c in self.edges.values() if c.enabled]
            if not enabled_edges:
                return

            edge_to_split = choice(enabled_edges)
            self.add_node(edge_to_split.id)

    def add_edge_mutation(self):
        """Applies the 'add edge' mutation."""
        if random() < self.config.add_edge_mutation_prob and len(self.nodes) >= 2:
            max_try = 35  # attempt to find a valid new connection
            for _ in range(max_try):
                node1 = choice(list(self.nodes.values()))
                node2 = choice(list(self.nodes.values()))
                if node1.id == node2.id:
                    continue
                # try adding edge n1 -> n2
                if self.add_edge(node1.id, node2.id):
                    break

    def remove_node_mutation(self):
        """Applies the 'remove node' mutation."""
        if random() < self.config.remove_node_mutation_prob:
            node_ids = [
                node_id
                for node_id in self.nodes
                if self.get_node_degree(node_id) == 2
            ]
            if node_ids:
                self.remove_node(choice(node_ids))

    def remove_edge_mutation(self):
        """Applies the 'remove edge' mutation."""
        if random() < self.config.remove_edge_mutation_prob and self.edges:
            edge_id_to_remove = choice(list(self.edges.keys()))
            self.remove_edge(edge_id_to_remove)

    def remove_node_single_mutation(self):
        """Applies the 'remove node (degree-1)' mutation."""
        if random() < self.config.remove_node_single_mutation_prob:
            node_ids = [
                node_id
                for node_id in self.nodes
                if self.get_node_degree(node_id) == 1
            ]
            if node_ids:
                self.remove_node_single(choice(node_ids))

    def mutate(self):
        """Applies all standard and specialized mutations."""
        self.add_node_mutation()
        self.add_edge_mutation()
        self.remove_edge_mutation()
        self.remove_node_mutation()
        self.remove_node_single_mutation()
        self.specialization.mutate(nodes=self.nodes, edges=self.edges)

    def get_node_degree(self, node_id: NodeID) -> int:
        return sum(
            1
            for edge in self.edges.values()
            if edge.enabled and (edge.node1 == node_id or edge.node2 == node_id)
        )

    def perform_remove_node(self, node_id: NodeID):
        connected_edges = [
            edge for edge in self.edges.values()
            if edge.node1 == node_id or edge.node2 == node_id
        ]
        enabled_edges = [edge for edge in connected_edges if edge.enabled]
        if len(enabled_edges) != 2:
            return
        edge1, edge2 = enabled_edges
        neighbor1 = edge1.node2 if edge1.node1 == node_id else edge1.node1
        neighbor2 = edge2.node2 if edge2.node1 == node_id else edge2.node1
        new_edge_id = self.population.get_edge_id(neighbor1, neighbor2)
        new_edge = Edge(node1=neighbor1, node2=neighbor2, id=new_edge_id)
        node = self.nodes[node_id]

        if not self.specialization.remove_node(
            node=node,
            edge1=edge1,
            edge2=edge2,
            new_edge=new_edge,
            nodes=self.nodes,
            edges=self.edges,
        ):
            return

        for edge in connected_edges:
            self.edges.pop(edge.id, None)
        self.nodes.pop(node_id, None)
        self.edges[new_edge.id] = new_edge

    def perform_remove_node_single(self, node_id: NodeID):
        connected_edges = [
            edge for edge in self.edges.values()
            if edge.node1 == node_id or edge.node2 == node_id
        ]
        enabled_edges = [edge for edge in connected_edges if edge.enabled]
        if len(enabled_edges) != 1:
            return
        edge = enabled_edges[0]
        node = self.nodes[node_id]
        if not self.specialization.remove_node_single(
            node=node,
            edge=edge,
            nodes=self.nodes,
            edges=self.edges,
        ):
            return
        for edge in connected_edges:
            self.edges.pop(edge.id, None)
        self.nodes.pop(node_id, None)

    def print_graph(self):
        """Prints a simple text representation of the genome's graph."""
        print("--- Genome Graph ---")
        print(f"Fitness: {self.fitness:.4f}")
        print("Nodes:")
        for node in self.sorted_nodes:
            print(f"  {node}")
        print("Edges:")
        for edge in self.sorted_edges:
            print(f"  {edge}")
        print("--------------------\n")

class Species:
    """
    Represents a species: a group of genetically similar genomes.
    
    Speciation protects innovation by allowing genomes to compete
    primarily within their niche.
    """
    def __init__(self, representative: Genome, config: NEATCoreConfig):
        self.config = config
        self.members = [representative]
        self.representative = representative
    
    def add_member(self, genome: Genome):
        self.members.append(genome)

    def adjust_fitness(self):
        """
        Applies explicit fitness sharing by dividing each member's
        fitness by the species size.
        """
        n = len(self.members)
        if n>0:
            for genome in self.members:
                genome.adj_fitness /= n

    def sorted(self, reverse=True) -> List[Genome]:
        """Returns members sorted by fitness, descending."""
        return sorted(self.members, key=lambda g: g.fitness, reverse=reverse)

class Population:
    """
    Manages the state of the population (genomes, IDs, species).
    All algorithmic logic (speciation, selection) has been moved to standalone functions.
    """
    def __init__(self, config: NEATCoreConfig, specialization: type[NEATSpecialization], start_genome: Optional[Genome] = None):
        self.config: NEATCoreConfig = config
        self.specialization: NEATSpecialization = specialization(config=self.config)
        self.dynamic_species_threshold = config.species_threshold
        self.species: List[Species] = []
        self.edge_id: EdgeID = -1
        self.node_id: NodeID = -1
        
        # tracks innovation IDs for (node1, node2) pairs
        self.edge_genes: Dict[Tuple[NodeID, NodeID], EdgeID] = {}
        self.members: List[Genome] = []

        self._initialize(start_genome)

    def get_edge_id(self, node1_id: NodeID, node2_id: NodeID) -> EdgeID:
        """
        Gets or assigns a unique innovation ID for an edge topology.
        """
        key = self.specialization.normalize_edge(node1_id, node2_id)
        existing_id = self.edge_genes.get(key)

        if existing_id is not None:
            return existing_id
        else:
            # new edge topology
            self.edge_id += 1
            self.edge_genes[key] = self.edge_id
            return self.edge_id

    def new_node_id(self) -> NodeID:
        """Assigns and returns a new unique node ID."""
        self.node_id += 1
        return self.node_id

    def _initialize(self, start_genome: Optional[Genome]):
        """Creates the initial population."""
        if start_genome is None:
            # create a template genome which updates id counters
            template_genome = Genome(
                config=self.config, 
                specialization=self.specialization,
                population=self, 
            )
            self.members = [
                template_genome.copy()
                for _ in range(self.config.population_size)
            ]
            for genome in self.members:
                genome.mutate()
        else:
            # initialize from an existing genome
            start_genome_max_node_id = max(n.id for n in start_genome.nodes.values())
            self.node_id = max(self.node_id, start_genome_max_node_id)
            
            start_genome_max_edge_id = max(c.id for c in start_genome.edges.values())
            self.edge_id = max(self.edge_id, start_genome_max_edge_id)

            # re-populate the edge_genes tracker
            for edge in start_genome.edges.values():
                key = self.specialization.normalize_edge(edge.node1, edge.node2)
                self.edge_genes[key] = edge.id

            for _ in range(self.config.population_size):
                genome = start_genome.copy()
                genome.population = self
                genome.fitness = 0
                self.members.append(genome)

    def get_top_genomes(self, k=1) -> List[Genome]:
        """
        Returns the k-best genomes.
        """
        if not self.members:
             raise IndexError("Cannot get top genome from an empty population.")

        return sorted(self.members, key=lambda g: g.fitness, reverse=True)[:k]

    def get_initial_genomes(self) -> List[Genome]:
        return self.members

    def get_stats(self) -> Tuple[float, float, int, List[float]]:
        if not self.members:
            return 0.0, 0.0, len(self.species), []
        max_fit = max(g.fitness for g in self.members)
        avg_fit = sum(g.fitness for g in self.members) / len(self.members)
        species_fits = [
            max((g.fitness for g in s.members), default=0.0)
            for s in self.species
        ]
        return max_fit, avg_fit, len(self.species), species_fits

    @property
    def current_species_threshold(self) -> float:
        return self.dynamic_species_threshold


# --- algorithmic functions (the toolkit) ---

def categorize_genes(
        genome1: Genome, genome2: Genome
    ) -> Dict[str, Tuple[List[Edge], List[Edge], List[Edge]]]:
    """
    Compares two genomes and categorizes their edge genes into
    matching, disjoint, and excess.
    """
    genes1 = genome1.sorted_edges
    genes2 = genome2.sorted_edges

    matching1, matching2 = [], []
    disjoint1, disjoint2 = [], []
    excess1, excess2 = [], []

    max_id1 = genes1[-1].id if genes1 else -1
    max_id2 = genes2[-1].id if genes2 else -1

    idx1, idx2 = 0, 0
    while idx1 < len(genes1) or idx2 < len(genes2):
        edge1 = genes1[idx1] if idx1 < len(genes1) else None
        edge2 = genes2[idx2] if idx2 < len(genes2) else None

        id1 = edge1.id if edge1 else float('inf')
        id2 = edge2.id if edge2 else float('inf')

        if id1 == id2:  # matching genes
            matching1.append(edge1)
            matching2.append(edge2)
            idx1 += 1
            idx2 += 1
        elif id1 < id2:  # gene in genome1 only
            if id1 > max_id2:
                    excess1.append(edge1)
            else:
                    disjoint1.append(edge1)
            idx1 += 1
        elif id2 < id1:  # gene in genome2 only
            if id2 > max_id1:
                excess2.append(edge2)
            else:
                disjoint2.append(edge2)
            idx2 += 1

    return {
        'genome1': (matching1, disjoint1, excess1),
        'genome2': (matching2, disjoint2, excess2)
    }

def calculate_compatibility(pop: Population, genome1: Genome, genome2: Genome) -> float:
    """
    Calculates the compatibility distance (delta) between two genomes.
    
    delta = (c1 * E / N) + (c2 * D / N)
    """
    categorized = categorize_genes(genome1, genome2)
    matching1, disjoint1, excess1 = categorized['genome1']
    matching2, disjoint2, excess2 = categorized['genome2']

    n1 = len(genome1.edges)
    n2 = len(genome2.edges)
    N = max(1.0, float(max(n1, n2)))  # genes in larger genome

    E = float(len(excess1) + len(excess2))
    D = float(len(disjoint1) + len(disjoint2))

    c1 = pop.config.c1
    c2 = pop.config.c2

    delta = (c1 * E / N) + (c2 * D / N) + pop.specialization.compatibility(
        genome1, matching1, disjoint1, excess1,
        genome2, matching2, disjoint2, excess2,
    )
    return delta

def crossover(pop: Population, genome1: Genome, genome2: Genome) -> Genome:
    """
    Performs crossover between two parent genomes.
    
    Offspring inherits all genes from the fitter parent. For matching
    genes, one is chosen randomly.
    """
    # ensure genome1 is the fitter parent
    if genome2.fitness > genome1.fitness:
        genome1, genome2 = genome2, genome1

    if not genome1.config.crossover_enabled:
        return genome1.copy()

    categorized = categorize_genes(genome1, genome2)
    matching1, disjoint1, excess1 = categorized['genome1'] # genes from fitter
    matching2, _, _ = categorized['genome2'] # matching genes from less fit

    offspring_nodes = {node.id: node.copy() for node in genome1.nodes.values()}
    offspring_edges = {}

    # process matching genes: choose randomly, but always use fitter parent's enabled status
    for edge1, edge2 in zip(matching1, matching2):
        chosen_edge_gene = choice((edge1, edge2)).copy()
        chosen_edge_gene.enabled = edge1.enabled  # edge1 is from the fitter parent
        offspring_edges[chosen_edge_gene.id] = chosen_edge_gene

    # inherit disjoint and excess genes from fitter parent
    for edge in disjoint1 + excess1:
        new_edge = edge.copy()
        offspring_edges[new_edge.id] = new_edge
    
    offspring = Genome(
        population=pop, 
        edges=offspring_edges, 
        nodes=offspring_nodes, 
        config=pop.config,
        specialization=pop.specialization,
    )

    return offspring

def tournament_select(candidates: List[Genome], k=2) -> Genome:
    pool = [choice(candidates) for _ in range(k)]
    return max(pool, key=lambda g: g.fitness)

def make_offspring_global_tournament(pop: Population, n_offspring: int) -> Tuple[List[Genome], List[Genome]]:
    elites = [g.copy() for g in pop.get_top_genomes(pop.config.num_elites_global)]
    children = []
    for _ in range(n_offspring-len(elites)):
        p1 = tournament_select(pop.members, k=pop.config.tournament_size)
        p2 = tournament_select(pop.members, k=pop.config.tournament_size)
        children.append(crossover(pop, p1, p2))
    return elites, children

# --- policies namespaces ---

class NEATSpeciation:
    """Namespace for Standard NEAT policies (Speciation, Sharing, Offspring)."""

    @staticmethod
    def respeciate(pop: Population):
        """
        Assigns genomes to species and updates the adaptive threshold.
        """
        # clear old members from species
        for s in pop.species:
            s.members = []
        
        # assign genomes to species
        for genome in pop.members:
            found = False
            for s in pop.species:
                delta = calculate_compatibility(pop, s.representative, genome)
                if delta < pop.dynamic_species_threshold:
                    s.add_member(genome)
                    found = True
                    break
            
            if not found:
                pop.species.append(Species(representative=genome, config=pop.config))
        
        # remove empty species
        pop.species = [s for s in pop.species if s.members]
        
        # update representatives for next generation (random member)
        for s in pop.species:
            s.representative = choice(s.members)

        if pop.config.target_species is not None:
            num_species = len(pop.species)
            difference = pop.config.target_species - num_species
            delta = difference / 100.0
            pop.dynamic_species_threshold = max(
                0.01,
                min(100.0, pop.dynamic_species_threshold - delta)
            )
        
        # # adjust threshold
        # num_species = len(pop.species)
        # if pop.config.adaptive_threshold > 0.0:
        #     if num_species > pop.config.target_species_number:
        #         pop.dynamic_species_threshold += pop.config.adaptive_threshold
        #     elif num_species < pop.config.target_species_number:
        #         pop.dynamic_species_threshold -= pop.config.adaptive_threshold

        #     pop.dynamic_species_threshold = max(
        #         pop.config.min_species_threshold, 
        #         min(pop.dynamic_species_threshold, pop.config.max_species_threshold)
        #     )

    @staticmethod
    def make_offspring(pop: Population, num_offspring: int) -> Tuple[List[Genome], List[Genome]]:
        # global elitism
        elites = [g.copy() for g in pop.get_top_genomes(pop.config.num_elites_global)]
        children = []
        species_offspring = num_offspring - pop.config.num_elites_global

        if pop.members:
            valid_members = [g for g in pop.members if g.fitness > -float('inf')]
            if not valid_members:
                min_fit = 0.0
                max_fit = 0.0
            else:
                min_fit = min(g.fitness for g in valid_members)
                max_fit = max(g.fitness for g in valid_members)
                
            rng = max(max_fit - min_fit, 1e-7)
            baseline = rng * 0.1
            offset = -min_fit + baseline
            for g in pop.members:
                if g.fitness == -float('inf'):
                    g.adj_fitness = 0.0
                else:
                    g.adj_fitness = g.fitness + offset
        else:
            for g in pop.members:
                g.adj_fitness = g.fitness

        avg_fits = []
        for s in pop.species:
            s.adjust_fitness() # divides g.adj_fitness by species size
            total = sum(g.adj_fitness for g in s.members)
            avg_fits.append(total)
        
        total_avg = sum(avg_fits)
        if total_avg == 0: total_avg = 1.0
        
        for s, avg in zip(pop.species, avg_fits):
            count = int(round((avg / total_avg) * species_offspring))
            if count <= 0: continue
            
            s_members = s.sorted()
            
            # species elitism
            n_elites = min(pop.config.num_elites_species, len(s_members), count)
            for i in range(n_elites):
                elites.append(s_members[i].copy())
                count -= 1
            if count <= 0: continue
            
            selection_pool = s_members[:max(1, int(len(s_members) * pop.config.selection_share))]
            if not selection_pool: continue
            
            for _ in range(count):
                p1 = choice(selection_pool)
                p2 = choice(selection_pool)
                children.append(crossover(pop, p1, p2))

        # fix rounding errors (fill remaining)
        while len(elites)+len(children) < num_offspring:
            s = choice(pop.species)
            if not s.members: continue
            p1 = choice(s.members)
            p2 = choice(s.members)
            children.append(crossover(pop, p1, p2))
        while len(elites)+len(children) > num_offspring:
            children.remove(choice(children))

        return elites, children

    @staticmethod
    def select(pop: Population, elites: List[Genome], children: List[Genome]):
        pop.members = elites + children
        NEATSpeciation.respeciate(pop)

# --- main interface ---

@dataclass
class GenerationStatistics:
    generation: int
    time_sec: float
    max_fitness: float
    avg_fitness: float

@dataclass
class RunStatistics:
    config: NEATCoreConfig
    start_time_sec: float
    gens: List[GenerationStatistics] = field(default_factory=list)

    def record_gen(self, gen: int, pop: Population, print_: bool = True):
        fitnesses = [genome.fitness for genome in pop.members]
        if not fitnesses: return

        top_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        gen_stats = GenerationStatistics(
            generation=gen,
            max_fitness=top_fitness,
            avg_fitness=avg_fitness,
            time_sec=time.time() - self.start_time_sec,
        )
        self.gens.append(gen_stats)

        if print_:
            species_number = len(pop.species)
            # print top species info if we are using speciation
            performers_str = ""
            if species_number > 0:
                 avg = lambda ls: sum(ls)/len(ls)
                 species_avg_fitnesses = {
                     s: avg([genome.fitness for genome in s.members if genome.fitness is not None])
                     for s in pop.species
                 }
                 species_sorted = sorted(species_avg_fitnesses.items(), key=lambda item: item[1], reverse=True)
                 fits = [s.sorted(reverse=True)[0].fitness for s,_ in species_sorted if s.members]
                 performers_str = " | top species: " + ', '.join(f'{fit:08.4f}' for fit in fits[:5])

            print(f"Gen {str(gen+1).zfill(4)} | "+
                  f"Avg fit {avg_fitness:08.4f} | "+
                  f"Top fit {top_fitness:08.4f} | "+
                  f"Species {species_number}" + performers_str)
    
    def record_final(self, top_genomes: List[Genome], print_: bool = True):
        if print_:
            print("\n--- Evolution Finished ---")
            if top_genomes:
                print(f"Top genome fitness: {top_genomes[-1].fitness}")
                print("Top graph:")
                top_genomes[-1].print_graph()
            else:
                print("No top genomes recorded.")

def _precompute_threshold(
    config: NEATCoreConfig,
    f_eval: Callable[[List[Genome]], List[float]],
    specialization_cls,
    target_species: int,
    gens: int = 15,
    trials: int = 10,
    print_: bool = True,
) -> float:
    """
    Performs short runs to find a species threshold that yields roughly `target_species`.
    Returns the calibrated threshold.
    """
    if print_:
        print(f"--- Precomputing Threshold for {target_species} Species ---")

    low = 0.1
    high = 100.0

    current = max(low, min(high, config.species_threshold))

    best_thresh = current
    min_diff = float('inf')

    for i in range(trials):
        temp_cfg = deepcopy(config)
        temp_cfg.species_threshold = current

        pop = Population(temp_cfg, specialization_cls)

        def evaluate(genomes):
            raw_evals = f_eval(genomes)
            if len(raw_evals) == 0:
                return
            for g, f in zip(genomes, raw_evals):
                g.fitness = f

        evaluate(pop.members)
        NEATSpeciation.respeciate(pop)

        final_species = 0
        for _ in range(gens):
            elites, children = NEATSpeciation.make_offspring(pop, temp_cfg.population_size)
            for child in children:
                child.mutate()
            offspring = elites + children
            evaluate(offspring)
            NEATSpeciation.select(pop, elites, children)
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


def run(
    config: NEATCoreConfig,
    f_eval: Callable[[List[Genome]], List[float]],
    specialization_cls,
    print_: bool = True,
) -> Tuple[List[Genome], Population, RunStatistics]:
    """
    Runs the neat algorithm.
    This function acts as a factory, wiring up the pipeline functions based on config.
    """
    ngen = config.ngen if config.ngen is not None else None
    target_fit = config.target_fit
    start_genome = config.start_genome
    precompute_threshold = config.precompute_threshold

    assert ngen is not None or target_fit is not None

    if precompute_threshold:
        new_thresh = _precompute_threshold(config, f_eval, specialization_cls, config.target_species, print_=print_)
        config.species_threshold = new_thresh

    pop = Population(config, specialization_cls, start_genome=start_genome)
    stats = RunStatistics(config, start_time_sec=time.time())
    top_genomes = []

    if ngen is None: ngen = int(1e6)

    def evaluate(genomes):
        raw_evals = f_eval(genomes)
        if len(raw_evals) == 0:
            return
        for g,f in zip(genomes,raw_evals):
            g.fitness = f

    n_offspring = config.population_size

    # get initial evaluations to enable mating
    evaluate(pop.members)

    if config.mating_strategy == MatingStrategy.NEATSpeciation:
        NEATSpeciation.respeciate(pop)

    for gen in range(ngen):
        stats.record_gen(gen, pop, print_)
        assert len(pop.members) == config.population_size, f"wrong population size: {len(pop.members)}"

        # record best genome and check stopping criterion
        g = pop.get_top_genomes()[0]
        top_genomes.append(g)
        if target_fit is not None and g.fitness >= target_fit:
            if print_: print(f"\nTarget fitness {target_fit} reached.")
            break

        # create offspring
        if config.mating_strategy == MatingStrategy.NEATSpeciation:
            elites, children = NEATSpeciation.make_offspring(pop, n_offspring)
        elif config.mating_strategy == MatingStrategy.GlobalTournament:
            elites, children = make_offspring_global_tournament(pop, n_offspring)

        # mutate offspring (not elites!)
        for child in children:
            child.mutate()

        offspring = elites + children
        evaluate(offspring)

        # select new population
        if config.mating_strategy == MatingStrategy.NEATSpeciation:
            NEATSpeciation.select(pop, elites, children)
        elif config.mating_strategy == MatingStrategy.GlobalTournament:
            pool = pop.members + elites + children
            pool.sort(key=lambda g: g.fitness, reverse=True)
            pop.members = pool[:config.population_size]
            pop.species = [] # No longer valid

    stats.record_final(top_genomes, print_=print_)
    return top_genomes, pop, stats
