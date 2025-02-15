import networkx as nx
import matplotlib.pyplot as plt
from generating_graphs import generate_planar_graph
from tqdm import tqdm

def experiment_planarity_threshold(num_nodes=20, max_edges=100, trials_per_edge=10000):
    """ Experiment to find the threshold for planarity """

    # Theoretical limit for planarity 
    theoretical_limit = 3 * num_nodes - 6
    print(f"Theoretical limit for planarity: {theoretical_limit} edges")
    
    # Run the experiment
    results = []
    for num_edges in tqdm(range(num_nodes - 1, max_edges + 1), desc="Testing edges", unit="edge"):
        planar_count = 0
        
        for _ in range(trials_per_edge):
            G, _ = generate_planar_graph(num_nodes, num_edges)
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

results = experiment_planarity_threshold()
plot_experiment(results)