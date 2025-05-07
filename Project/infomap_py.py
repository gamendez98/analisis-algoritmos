import pickle
import random
from collections import defaultdict
import networkx as nx
from matplotlib import pyplot as plt

from Project.game_theory import GameTheoreticCommunityDetection
from Project.infomap_impl import Infomap
from Project.infomap_nx_adapter import nx_infomap_partition
import time


def compute_stationary_distribution(G, teleportation=0.15, max_iter=100, tol=1e-8):
    """
    Compute stationary distribution via power method with teleportation.

    Returns:
      dict mapping node -> probability.
    Complexity: O(max_iter * (N + E))
    """
    assert 0 < teleportation < 1, "Teleportation must be in (0, 1)"
    N = G.number_of_nodes()

    # Initialize with a uniform distribution
    p = {node: 1.0 / N for node in G.nodes()}

    for _ in range(max_iter):
        # Start with the teleportation contribution for every node.
        new_p = {node: teleportation / N for node in G.nodes()}
        # For each node, distribute its probability mass along its outgoing edges.
        for node in G.nodes():
            # Get all outgoing edges from node along with their weight.
            out_edges = list(G.out_edges(node, data=True))
            if not out_edges:
                # Dangling node: no outlinks. Distribute its (follow) probability uniformly.
                contribution = (1 - teleportation) * p[node] / N
                for dest in G.nodes():
                    new_p[dest] += contribution
            else:
                # Total weight of node's out-going links.
                total_w = sum(data.get('weight', 1) for (_, _, data) in out_edges)
                for (_, neighbor, data) in out_edges:
                    weight = data.get('weight', 1)
                    new_p[neighbor] += (1 - teleportation) * p[node] * (weight / total_w)

        # Check convergence (L1 norm of difference)
        err = sum(abs(new_p[n] - p[n]) for n in G.nodes())
        p = new_p
        if err < tol:
            break

    sum_p = sum(p.values())
    assert abs(sum_p - 1.0) < 1e-6, f"Stationary dist not normalized: sum={sum_p}"
    return p


def simulate_random_walks(graph, num_walks=1000, walk_length=10):
    visits = defaultdict(int)
    transitions = defaultdict(lambda: defaultdict(int))

    nodes = list(graph.nodes())
    for _ in range(num_walks):
        node = random.choice(nodes)
        for _ in range(walk_length):
            visits[node] += 1
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            transitions[node][next_node] += 1
            node = next_node
    return visits, transitions

def map_equation(communities, transitions, visits, total_visits):
    """Simplified map equation"""
    q = 0  # inter-community transitions
    L = 0  # total description length
    for c in set(communities.values()):
        nodes_in_c = [n for n in communities if communities[n] == c]
        p_in = sum(visits[n] for n in nodes_in_c) / total_visits
        p_exit = 0
        for u in nodes_in_c:
            for v in transitions[u]:
                if communities[v] != c:
                    p_exit += transitions[u][v] / total_visits
        if p_in + p_exit > 0:
            L -= (p_in + p_exit) * log2(p_in + p_exit)
        if p_in > 0:
            L += p_in * log2(p_in)
        q += p_exit
    L += q * log2(q) if q > 0 else 0
    return L

def log2(x):
    import math
    return math.log(x, 2)

def infomap_basic(graph):
    # Step 1: Simulate walks
    visits, transitions = simulate_random_walks(graph, walk_length= 30 * graph.number_of_nodes(), num_walks=1000)
    total_visits = sum(visits.values())

    # Step 2: Initialize each node to its own community
    communities = {node: node for node in graph.nodes()}
    current_cost = map_equation(communities, transitions, visits, total_visits)

    # Step 3: Try merging communities greedily
    improved = True
    while improved:
        improved = False
        for node in graph.nodes():
            best_comm = communities[node]
            best_cost = current_cost
            neighbor_comms = {communities[neigh] for neigh in graph.neighbors(node)}

            for comm in neighbor_comms:
                if comm == best_comm:
                    continue
                original = communities[node]
                communities[node] = comm
                new_cost = map_equation(communities, transitions, visits, total_visits)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_comm = comm
                    improved = True
                communities[node] = original  # revert

            communities[node] = best_comm
            current_cost = best_cost
    return communities

def display_partition(partition):
    g = GameTheoreticCommunityDetection(G)
    g.partition = partition
    g.visualize()
    plt.show()

if __name__ == "__main__":
    with open('test_graphs/benchmark_graph_1000_4.pkl', 'rb') as f:
        G = pickle.load(f)
    # with open('test2/benchmark_graph_250_3.pkl', 'rb') as f:
    #     G = pickle.load(f)
    #G = nx.connected_caveman_graph(10, 10)

    # comms = infomap_basic(G)
    # display_partition(comms)
    #part = nx_infomap_partition(G)
    #display_partition(part)
    start = time.time()
    part = GameTheoreticCommunityDetection(G).run()[0]
    #part = Infomap(nx.DiGraph(G)).run()[0]
    end = time.time()
    print(end - start)
    display_partition(part)
