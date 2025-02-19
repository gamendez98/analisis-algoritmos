import random
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from datetime import datetime

def generate_planar_graph(num_nodes, num_edges, max_trials=1000, ensure_connectivity=False, verbose = False):
    """  Generates a random planar graph with num_nodes nodes and num_edges edges. """

    # Check if the required number of edges is sufficient for connectivity
    # Raises Value Error if not.
    if num_edges < num_nodes - 1:
        raise ValueError("The number of edges must be at least num_nodes - 1 to ensure connectivity.")
    
    # Generate random points
    points = [(random.random(), random.random()) for _ in range(num_nodes)]
    
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            edges.add(edge)
    
    # Edges to list and pick the ones required
    edges = list(edges)
    random.shuffle(edges)
    G = nx.Graph()
    for i, (x, y) in enumerate(points):
        G.add_node(i, pos=(x, y))
    G.add_edges_from(edges[:min(num_edges, len(edges))])
    
    # Ensure we reach the required number of edges while keeping planarity
    trials = 0
    attempted_edges = set()
    all_possible_edges = {(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)}

    while G.number_of_edges() < num_edges and trials < max_trials:
        if attempted_edges == all_possible_edges:
            if verbose:
                print("Warning: Exhausted all possible edges without reaching the required number.")
            break
        
        u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        edge = tuple(sorted([u, v]))
        
        if u != v and edge not in attempted_edges:
            attempted_edges.add(edge)
            G.add_edge(u, v)
            if not nx.is_planar(G):
                G.remove_edge(u, v)
        trials += 1
    
    if G.number_of_edges() < num_edges:
        if verbose:
            print(f"Warning: Could not reach the desired number of edges within {max_trials} trials.")
    
    # Ensure the graph is connected
    if ensure_connectivity and not nx.is_connected(G):
        if verbose:
            print("Warning: The generated graph is not connected. Adding edges to ensure connectivity.")
        components = list(nx.connected_components(G))
        while len(components) > 1:
            comp1, comp2 = components[0], components[1]
            node1, node2 = random.choice(list(comp1)), random.choice(list(comp2))
            G.add_edge(node1, node2)
            components = list(nx.connected_components(G))
    
    return G, points

def plot_graph(G):
    """ Plots the graph G with a planar layout if possible """
    pos = nx.planar_layout(G) if nx.check_planarity(G)[0] else nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
    plt.title("Random Planar Graph")
    plt.show()

def save_graph_csv(G, num_nodes):
    """ Saves the graph as a CSV file """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"planar_graph_{num_nodes}nodes_{G.number_of_edges()}edges_{timestamp}.csv"
    with open(filename, "w") as f:
        f.write("Source,Target\n")
        for edge in G.edges():
            f.write(f"{edge[0]},{edge[1]}\n")
    print(f"Graph saved as {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random planar graph.")
    parser.add_argument("num_nodes", type=int, help="Number of nodes in the graph")
    parser.add_argument("num_edges", type=int, help="Number of edges in the graph")
    parser.add_argument("--max_trials", type=int, default=1000, help="Maximum number of trials for edge placement (default: 1000)")
    parser.add_argument("--ensure_connectivity", action="store_true", help="Ensure the graph is connected")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    G, points = generate_planar_graph(args.num_nodes, args.num_edges, max_trials=args.max_trials, ensure_connectivity=args.ensure_connectivity, verbose=args.verbose)
    
    is_planar, _ = nx.check_planarity(G)
    print(f"Is the generated graph planar? {is_planar}")
    print(f"The generated graph has {G.number_of_edges()} edges.")
    save_graph_csv(G, args.num_nodes)
    plot_graph(G)