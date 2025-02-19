import argparse
import networkx as nx
import matplotlib.pyplot as plt
from generating_graphs import generate_planar_graph
from tqdm import tqdm

def experiment_planarity_threshold(num_nodes=20, max_edges=70, trials_per_edge=1000):
    """ Experiment to find the threshold for planarity """

    # Theoretical limit for planarity 
    theoretical_limit = 3 * num_nodes - 6
    print(f"Theoretical limit for planarity: {theoretical_limit} edges")
    
    # Run the experiment
    results = []
    for num_edges in tqdm(range(num_nodes - 1, max_edges + 1), desc="Testing edges", unit="edge"):
        planar_count = 0
        
        for _ in range(trials_per_edge):
            G, _ = generate_planar_graph(num_nodes, num_edges, verbose=False)
            if G.number_of_edges() == num_edges and nx.is_planar(G):
                planar_count += 1
        
        probability_planar = planar_count / trials_per_edge
        results.append((num_edges, probability_planar))
        print(f"Edges: {num_edges}, Probability of being planar: {probability_planar:.2f}")
    
    return results

def plot_experiment(results):
    """ Plots the results of the planarity experiment """
    edges, probabilities = zip(*results)
    plt.figure(figsize=(8, 6))
    plt.plot(edges, probabilities, marker='o', linestyle='-')
    plt.xlabel("Number of Edges")
    plt.ylabel("Probability of being Planar")
    plt.title("Planarity Experiment for 20 Nodes")
    plt.axvline(x=3 * 20 - 6, color='r', linestyle='--', label="Theoretical Limit")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a planarity threshold experiment.")
    parser.add_argument("--num_nodes", type=int, default=20, help="Number of nodes in the graph (default: 20)")
    parser.add_argument("--max_edges", type=int, default=70, help="Maximum number of edges to test (default: 70)")
    parser.add_argument("--trials_per_edge", type=int, default=1000, help="Number of trials per edge count (default: 1000)")
    args = parser.parse_args()
    
    results = experiment_planarity_threshold(args.num_nodes, args.max_edges, args.trials_per_edge)
    plot_experiment(results, args.num_nodes)