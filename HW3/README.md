# Planarity Experiment and Graph Generation

The homework includes two Python scripts that generate planar graphs and conduct experiments to determine the planarity threshold. The scripts use command-line arguments for flexibility and provide visual and CSV outputs.

## Team Members

- **David Alejandro Fuquen Florez**
- **Isabella Martinez Martinez**
- **Gustavo Andrés Mendez Hernández**

## Requirements

Ensure Python 3 is installed, along with the following dependencies:

```bash
pip install networkx matplotlib scipy tqdm
```

## Scripts

### 1. `generate_graphs.py`

Generates a random planar graph based on the given number of nodes and edges. By default, it needs not to be connected.

#### **Inputs**
- `num_nodes` (int, required): Number of nodes in the graph.
- `num_edges` (int, required): Number of suggested edges in the graph.
- `--max_trials` (int, optional, default=1000): Maximum attempts to add edges while maintaining planarity.
- `--ensure_connectivity` (flag, optional): Ensures the generated graph is connected.
- `--verbose` (flag, optional): Enables detailed warnings in graph generation.

#### **Outputs**
- A visualization of the generated graph.
- A CSV file containing the graph edges (`planar_graph_<n_nodes>nodes_<n_edges>edges_<timestamp>.csv`).

#### **How to Run**
```bash
python generate_graphs.py <num_nodes> <num_edges> [--max_trials 1000] [--ensure_connectivity] [--verbose]
```
Example:
```bash
python generate_graphs.py 20 30 --ensure_connectivity --verbose
```

---

### 2. `planar_experiment.py`

Conducts an experiment to determine the probability of a graph remaining planar as edges increase. It may take a while to run.

#### **Inputs**
- `--num_nodes` (int, optional, default=20): Number of nodes in the graph.
- `--max_edges` (int, optional, default=70): Maximum number of edges to test.
- `--trials_per_edge` (int, optional, default=1000): Number of trials per edge count.

#### **Outputs**
- A probability plot showing the likelihood of graphs being planar as edges increase.

#### **How to Run**
```bash
python planar_experiment.py [--num_nodes 20] [--max_edges 70] [--trials_per_edge 1000]
```
Example:
```bash
python planar_experiment.py --num_nodes 25 --max_edges 80 --trials_per_edge 500
```

## Notes
- The `generate_graphs.py` script is required by `planar_experiment.py`.
- Outputs are saved in the same directory as the scripts.
- Adjusting `--trials_per_edge` in the experiment script affects computation time.

