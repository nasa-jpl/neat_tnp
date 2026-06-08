# NEATWork: Track Network Planning using NEAT-based Evolution of Graphs

This repository contains

* A generalization of the [NEAT genetic algorithm](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) that synthesizes its core mechanism of evolving graphs, without knowledge about neural networks (`neat_core.py`)
* A re-implementation of the original NEAT use case on top of the generalized algorithm (`neat.py`)
* A new use case of heuristically finding spatial networks, applied to the track planning of a lunar cargo transport system (`neatwork.optimizer_ref`)
* A high-performance implementation of the track network planning use case, written in Rust

A set of minimal examples can be found in `examples.ipynb`.

## Installation

```
pip install neatwork
```

Or to build the wheel from sources, run:

```
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release -i python3.12 -o /io/dist
```

## Development

```bash
pip install .[dev]
```

Now you should be able to run the notebooks. Project structure:

```
├── __init__.py
├── neatwork
│   ├── neat_core_py.py
│   ├── neatwork_py.py          # reference NEATWork implementation
│   ├── neatwork_rs.py          # high-performance NEATWork implementation
│   ├── neat.py                 # re-implementation of original NEAT
│   ├── neatwork_py_eval.py     # internal evaluation helpers
│   └── utils
│       ├── dem_utils.py
│       └── plotting.py
├── crates                      # Rust crates for neatwork_rs.py
│   ├── neat_core_rs
│   └── neatwork_lib
├── examples.ipynb              # canonical examples
├── pyproject.toml              # dependencies and package definition
└── README.md                   # this file
```
