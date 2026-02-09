# NEAT-TNP: Track Network Planning using NEAT-based Evolution of Graphs

This repository contains

* A generalization of the [NEAT genetic algorithm](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) that synthesizes its core mechanism of evolving graphs, without knowledge about neural networks (`neat_core.py`)
* A re-implementation of the original NEAT use case on top of the generalized algorithm (`neat.py`)
* A new use case of heuristically finding spatial networks, applied to the track planning of a lunar cargo transport system (`neat_tnp.py`)
* A high-performance implementation of the TNP use case, written in Rust

A set of minimal examples can be found in `examples.ipynb`.

## Installation

```
pip install neat_tnp
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
├── neat_tnp
│   ├── neat_py_tnp.py          # reference TNP implementation
│   ├── neat_py_core.py
│   ├── neat_py_neat.py         # re-implementation of original NEAT
│   ├── neat_rs_tnp.py          # high-performance TNP implementation
│   ├── plotting.py             # utilities
│   ├── config_utils.py
│   ├── dem_utils.py
│   └── neat_py_tnp_eval.py     
├── crates                  # Rust crates for neat_rs_tnp.py
│   ├── neat_rs_core
│   └── neat_rs_tnp_rs
├── examples.ipynb          # canonical examples
├── pyproject.toml          # dependencies and package definition
└── README.md               # this file
```
