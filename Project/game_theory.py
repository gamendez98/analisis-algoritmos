from collections import defaultdict
from itertools import product

import pandas as pd
import seaborn as sns
import networkx as nx
import numpy as np
import random
import time

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from Project.infomap_nx_adapter import nx_infomap_partition


class GameTheoreticCommunityDetection:
    """
    Community detection using game theory principles, where nodes act as
    agents maximizing their utility by joining the optimal community.
    """

    def __init__(self, G, internal_weight=1.0, external_penalty=2.0, size_penalty=0.1, max_iterations=100):
        """
        Initialize the game-theoretic community detector.

        Parameters:
            G (nx.Graph): Undirected graph with optional edge weights
            internal_weight (float): Weight for internal connectivity reward
            external_penalty (float): Weight for external connectivity penalty
            size_penalty (float): Penalty for large community sizes
            max_iterations (int): Maximum number of iterations
        """
        self.G = G
        self.internal_weight = internal_weight
        self.external_penalty = external_penalty
        self.size_penalty = size_penalty
        self.max_iterations = max_iterations
        self.partition = None
        self.next_label = 0

    def initialize_partition(self, method="singleton"):
        """
        Initialize communities using different strategies.

        Parameters:
            method (str): Initialization method ('random', 'singleton')

        Returns:
            dict: Node to community mapping
        """
        nodes = list(self.G.nodes())

        if method == "random":
            k = max(2, int(np.sqrt(len(nodes))))
            partition = {node: random.randint(0, k - 1) for node in nodes}

        elif method == "singleton":
            partition = {node: i for i, node in enumerate(nodes)}

        else:
            raise ValueError(f"Unknown initialization method: {method}")

        self.next_label = max(partition.values()) + 1 if partition else 0
        return partition

    def calculate_node_utility(self, node, community, partition, community_sizes):
        """
        Calculate utility for a node in a specific community, using edge weights.

        Parameters:
            node: The node to evaluate
            community: Target community ID
            partition: Current node->community mapping
            community_sizes: Pre computed community sizes

        Returns:
            float: Utility score
        """
        internal_edge_weights = 0
        external_edge_weights = 0

        for neighbor in self.G.neighbors(node):
            weight = self.G[node][neighbor].get('weight', 1.0)
            if partition.get(neighbor) == community:
                internal_edge_weights += weight
            else:
                external_edge_weights += weight

        # comm_size = sum(1 for n, c in partition.items() if c == community and n != node)
        comm_size = community_sizes[community] - (partition.get(node) == community)
        utility = (self.internal_weight * internal_edge_weights -
                   self.external_penalty * external_edge_weights -
                   self.size_penalty * comm_size)
        return utility

    @staticmethod
    def community_sizes_from_partition(partition):
        comm_sizes = defaultdict(int)
        for node, comm_id in partition.items():
            comm_sizes[comm_id] += 1
        return comm_sizes

    def find_best_community(self, node, partition, community_sizes):
        """
        Find the community that maximizes a node's utility, including option to form a singleton.

        Parameters:
            node: Node to evaluate
            partition: Current partition
            community_sizes: Pre computed community sizes

        Returns:
            tuple: (best community ID or 'new', utility)
        """
        current_community = partition.get(node)
        current_utility = self.calculate_node_utility(node, current_community, partition, community_sizes)
        best_utility = current_utility
        best_community = current_community

        candidate_communities = set()
        for neighbor in self.G.neighbors(node):
            if neighbor in partition:
                candidate_communities.add(partition[neighbor])

        for comm in candidate_communities:
            if comm != current_community:
                utility = self.calculate_node_utility(node, comm, partition, community_sizes)
                if utility > best_utility:
                    best_utility = utility
                    best_community = comm

        singleton_utility = -self.external_penalty * sum(
            self.G[node][neighbor].get('weight', 1.0) for neighbor in self.G.neighbors(node)
        )
        if singleton_utility > best_utility:
            best_utility = singleton_utility
            best_community = "new"

        return best_community, best_utility

    def force_split(self, partition):
        """
        If only one community exists, split it into two based on highest degree node and its neighbors.

        Parameters:
            partition: Current node->community mapping

        Returns:
            dict: Updated partition
            bool: Whether a split was performed
        """
        communities = set(partition.values())
        did_split = False
        if len(communities) > 1:
            return partition, did_split

        did_split = True

        degrees = self.G.degree(weight='weight')
        max_degree_node = max(degrees, key=lambda x: x[1])[0]

        new_partition = partition.copy()
        new_label = self.next_label
        self.next_label += 1

        new_partition[max_degree_node] = new_label
        for neighbor in self.G.neighbors(max_degree_node):
            new_partition[neighbor] = new_label

        return new_partition, did_split

    def merge_small_communities(self, partition, community_sizes, min_size=3):
        """
        Merge communities smaller than min_size into the best neighboring community.

        Parameters:
            partition: Current node->community mapping
            min_size: Minimum allowed community size
            community_sizes: Pre computed community sizes


        Returns:
            dict: Updated partition
            bool: Whether a merge was performed
        """
        communities = defaultdict(list)
        did_merge = False
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        new_partition = partition.copy()
        for comm_id, nodes in communities.items():
            if len(nodes) < min_size:
                did_merge = True
                best_comm = None
                best_utility = float('-inf')

                neighbor_comms = set()
                for node in nodes:
                    for neighbor in self.G.neighbors(node):
                        if neighbor in partition and partition[neighbor] != comm_id:
                            neighbor_comms.add(partition[neighbor])

                for target_comm in neighbor_comms:
                    total_utility = 0
                    for node in nodes:
                        total_utility += self.calculate_node_utility(node, target_comm, partition, community_sizes)
                    if total_utility > best_utility:
                        best_utility = total_utility
                        best_comm = target_comm

                if best_comm is not None:
                    for node in nodes:
                        new_partition[node] = best_comm

        return new_partition, did_merge

    def run(self, init_method="singleton"):
        """
        Execute the game-theoretic community detection algorithm.

        Parameters:
            init_method (str): Method to initialize communities

        Returns:
            dict: Final node to community mapping
        """
        partition = self.initialize_partition(method=init_method)

        iteration = 0
        changes = True

        community_sizes = self.community_sizes_from_partition(partition)

        while changes and iteration < self.max_iterations:
            changes = False
            nodes = list(self.G.nodes())
            random.shuffle(nodes)

            for node in nodes:
                best_community, _ = self.find_best_community(node, partition, community_sizes)
                current_community = partition.get(node)

                if best_community != current_community:
                    community_sizes[current_community] -= 1
                    if best_community == "new":
                        new_label = self.next_label
                        partition[node] = new_label
                        community_sizes[new_label] = 1
                        self.next_label += 1
                    else:
                        community_sizes[best_community] += 1
                        partition[node] = best_community
                    changes = True

            partition, did_force = self.force_split(partition)
            if did_force:
                community_sizes = self.community_sizes_from_partition(partition)
            partition, did_merge = self.merge_small_communities(partition, community_sizes, min_size=3)
            if did_merge:
                community_sizes = self.community_sizes_from_partition(partition)

            iteration += 1
            # print(f"Iteration {iteration}: {self.count_communities(partition)} communities")

        self.partition = self.renumber_communities(partition)
        return self.partition, iteration

    @staticmethod
    def count_communities(partition):
        """Count number of distinct communities in partition."""
        return len(set(partition.values()))

    @staticmethod
    def renumber_communities(partition):
        """Renumber community IDs to be consecutive integers starting from 0."""
        communities = defaultdict(list)
        for node, comm in partition.items():
            communities[comm].append(node)

        new_partition = {}
        for new_id, (_, nodes) in enumerate(communities.items()):
            for node in nodes:
                new_partition[node] = new_id
        return new_partition

    def calculate_modularity(self):
        """Calculate modularity of the current partition."""
        if self.partition is None:
            return None
        communities = defaultdict(list)
        for node, comm_id in self.partition.items():
            communities[comm_id].append(node)
        return nx.community.modularity(self.G, communities.values(), weight='weight')

    def calculate_community_metrics(self, ground_truth=None):
        """
        Calculate various metrics for the detected communities.

        Parameters:
            ground_truth (dict): Optional node to ground truth community mapping

        Returns:
            dict: Metrics including modularity, NMI, ARI, etc.
        """
        if self.partition is None:
            return None

        communities = defaultdict(list)
        for node, comm_id in self.partition.items():
            communities[comm_id].append(node)

        metrics = {
            'modularity': nx.community.modularity(self.G, communities.values(), weight='weight'),
            'num_communities': len(communities),
            'sizes': [len(nodes) for nodes in communities.values()],
            'internal_densities': [],
            'external_densities': [],
            'communities': communities
        }

        if ground_truth:
            true_labels = [ground_truth[node] for node in self.G.nodes()]
            detected_labels = [self.partition[node] for node in self.G.nodes()]
            metrics['nmi'] = normalized_mutual_info_score(true_labels, detected_labels)
            metrics['ari'] = adjusted_rand_score(true_labels, detected_labels)

        for comm_id, nodes in communities.items():
            subgraph = self.G.subgraph(nodes)
            potential_edges = len(nodes) * (len(nodes) - 1) / 2
            internal_edges = sum(subgraph[u][v].get('weight', 1.0) for u, v in subgraph.edges()) / 2
            internal_density = internal_edges / potential_edges if potential_edges > 0 else 0
            metrics['internal_densities'].append(internal_density)

            external_edges = 0
            for node in nodes:
                external_edges += sum(
                    self.G[node][neighbor].get('weight', 1.0)
                    for neighbor in self.G.neighbors(node)
                    if self.partition.get(neighbor) != comm_id
                )
            other_nodes = len(self.G.nodes()) - len(nodes)
            potential_external = len(nodes) * other_nodes
            external_density = external_edges / potential_external if potential_external > 0 else 0
            metrics['external_densities'].append(external_density)

        return metrics

    def visualize(self, with_labels=True, figsize=(12, 8), node_labels=None,
                  ground_truth=None,
                  ground_truth_title=None):
        """
        Visualize the graph with communities colored differently, optionally showing ground truth.

        Parameters:
            with_labels (bool): Whether to show node labels
            figsize (tuple): Figure size
            node_labels (dict): Optional mapping of node_id -> label
            ground_truth (dict): Optional ground truth partition for comparison
            ground_truth_title (str): Optional title for ground truth visualization
        """
        if self.partition is None:
            raise ValueError("Run the algorithm first")

        fig, ax = plt.subplots(1, 2 if ground_truth else 1,
                               figsize=(figsize[0] * (2 if ground_truth else 1), figsize[1]))
        if not ground_truth:
            ax = [ax]

        pos = nx.spring_layout(self.G, seed=42)

        communities = set(self.partition.values())
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(communities), 3)))

        for i, comm_id in enumerate(communities):
            nodes = [node for node, comm in self.partition.items() if comm == comm_id]
            nx.draw_networkx_nodes(self.G, pos, nodelist=nodes, node_color=[colors[i]],
                                   node_size=300, alpha=0.8, label=f"Community {comm_id}", ax=ax[0])

        nx.draw_networkx_edges(self.G, pos, alpha=0.5, ax=ax[0])
        if with_labels:
            nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=10, ax=ax[0])

        ax[0].set_title("Detected Communities")
        ax[0].legend(loc="upper right")
        ax[0].axis('off')

        if ground_truth:
            communities = set(ground_truth.values())
            for i, comm_id in enumerate(communities):
                nodes = [node for node, comm in ground_truth.items() if comm == comm_id]
                nx.draw_networkx_nodes(self.G, pos, nodelist=nodes, node_color=[colors[i]],
                                       node_size=300, alpha=0.8, label=f"True Community {comm_id}", ax=ax[1])

            nx.draw_networkx_edges(self.G, pos, alpha=0.5, ax=ax[1])
            if with_labels:
                nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=10, ax=ax[1])

            ax[1].set_title(ground_truth_title or "Ground Truth Communities")
            ax[1].legend(loc="upper right")
            ax[1].axis('off')

        plt.tight_layout()
        return plt


def test_benchmark_graph():
    # Load the benchmark graph
    G = nx.karate_club_graph()

    # Ground truth
    true_partition = {node: 0 if G.nodes[node]['club'] == 'Mr. Hi' else 1 for node in G.nodes()}

    # Define expanded parameter grid
    param_grid = {
        'internal_weight': [0.3, 0.5, 0.7],
        'external_penalty': [1.0, 1.5, 2.0],
        'size_penalty': [0.05, 0.1, 0.15, 0.2]
    }

    best_ari = -1
    best_params = None
    best_metrics = None
    best_detector = None

    # Number of runs per parameter set to account for randomness
    num_runs = 3

    # Grid search
    print("Running grid search...")
    for internal_weight, external_penalty, size_penalty in product(
            param_grid['internal_weight'], param_grid['external_penalty'], param_grid['size_penalty']
    ):
        print(
            f"\nTesting: internal_weight={internal_weight}, external_penalty={external_penalty}, size_penalty={size_penalty}")
        ari_scores = []
        metrics_list = []

        for run in range(num_runs):
            detector = GameTheoreticCommunityDetection(
                G, internal_weight=internal_weight, external_penalty=external_penalty, size_penalty=size_penalty
            )
            partition, _ = detector.run(init_method="singleton")
            metrics = detector.calculate_community_metrics(ground_truth=true_partition)
            ari_scores.append(metrics['ari'])
            metrics_list.append(metrics)

        # Average metrics over runs
        avg_ari = np.mean(ari_scores)
        avg_metrics = {
            'ari': avg_ari,
            'nmi': np.mean([m['nmi'] for m in metrics_list]),
            'modularity': np.mean([m['modularity'] for m in metrics_list]),
            'num_communities': int(np.mean([m['num_communities'] for m in metrics_list])),
            'sizes': metrics_list[0]['sizes'],  # Take sizes from last run (may vary slightly)
            'internal_densities': np.mean([np.mean(m['internal_densities']) for m in metrics_list]),
            'external_densities': np.mean([np.mean(m['external_densities']) for m in metrics_list])
        }

        print(f"Average ARI: {avg_ari:.4f}, NMI: {avg_metrics['nmi']:.4f}, Modularity: {avg_metrics['modularity']:.4f}")
        print(f"Average number of communities: {avg_metrics['num_communities']}, Example sizes: {avg_metrics['sizes']}")

        if avg_ari > best_ari:
            best_ari = avg_ari
            best_params = {
                'internal_weight': internal_weight,
                'external_penalty': external_penalty,
                'size_penalty': size_penalty
            }
            best_metrics = avg_metrics
            best_detector = detector  # Save the last detector for visualization

    # Report best results
    print("\nBest Parameters:")
    print(f"internal_weight: {best_params['internal_weight']}")
    print(f"external_penalty: {best_params['external_penalty']}")
    print(f"size_penalty: {best_params['size_penalty']}")
    print("\nBest Metrics (Averaged):")
    print(f"Number of communities: {best_metrics['num_communities']}")
    print(f"Example community sizes: {best_metrics['sizes']}")
    print(f"Modularity: {best_metrics['modularity']:.4f}")
    print(f"Normalized Mutual Information (NMI): {best_metrics['nmi']:.4f}")
    print(f"Adjusted Rand Index (ARI): {best_metrics['ari']:.4f}")
    print(f"Average internal density: {best_metrics['internal_densities']:.4f}")
    print(f"Average external density: {best_metrics['external_densities']:.4f}")

    # Visualize the best result
    best_detector.visualize(ground_truth=true_partition)
    plt.show()


def set_all_edge_weights(G, low, high):
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(low, high)


def test_run_time(l, k, max_iter=100):
    G = nx.connected_caveman_graph(l, k)
    set_all_edge_weights(G, 0.1, 1.0)
    detector = GameTheoreticCommunityDetection(G, max_iterations=max_iter)
    start_time = time.time()
    detector.run(init_method="singleton")
    end_time = time.time()
    run_time = end_time - start_time
    return G, run_time


def test_run_time_watts(n, k, p, max_iter=100):
    G = nx.connected_watts_strogatz_graph(n, k, p)
    set_all_edge_weights(G, 0.1, 1.0)
    detector = GameTheoreticCommunityDetection(G, max_iterations=max_iter)
    start_time = time.time()
    detector.run(init_method="singleton")
    end_time = time.time()
    run_time = end_time - start_time
    return G, run_time


def graph_run_time():
    run_times = []
    ns = []
    es = []
    for i in tqdm(range(1, 200)):
        G, run_time = test_run_time(i, 10)
        run_times.append(run_time)
        n = G.number_of_nodes()
        e = G.number_of_edges()
        es.append(e)
        ns.append(n)
    sns.lineplot(x=ns, y=run_times)
    sns.lineplot(x=es, y=run_times)
    df = pd.DataFrame({'n': ns, 'e': es, 'run_time': run_times})
    df.to_csv('run_times2.csv', index=False)
    plt.show()


def graph_run_time_watts():
    run_times = []
    ns = []
    es = []
    for i in tqdm(range(1, 200)):
        G, run_time = test_run_time_watts(i * 10, 10, 0.1)
        run_times.append(run_time)
        n = G.number_of_nodes()
        e = G.number_of_edges()
        es.append(e)
        ns.append(n)
    sns.lineplot(x=ns, y=run_times)
    sns.lineplot(x=es, y=run_times)
    df = pd.DataFrame({'n': ns, 'e': es, 'run_time': run_times})
    df.to_csv('run_times_watts2.csv', index=False)
    plt.show()


def graph_run_time_max_iter():
    max_iters = []
    run_times = []
    for i in tqdm(range(1, 300)):
        G, run_time = test_run_time_watts(4000, 10, 0.1, max_iter=i)
        run_times.append(run_time)
        max_iters.append(i)
    sns.scatterplot(x=max_iters, y=run_times)
    df = pd.DataFrame({'max_iters': max_iters, 'run_time': run_times})
    df.to_csv('run_times_max_iter2.csv', index=False)
    plt.show()


def test_iterations_needed_watts(n, k, p):
    G = nx.connected_watts_strogatz_graph(n, k, p)
    set_all_edge_weights(G, 0.1, 1.0)
    detector = GameTheoreticCommunityDetection(G, max_iterations=1000)
    start_time = time.time()
    _, iterations = detector.run(init_method="singleton")
    end_time = time.time()
    run_time = end_time - start_time
    return G, iterations


def graph_iterations_needed_watts():
    ns = []
    needed_iterations = []
    for i in tqdm(range(1, 300)):
        G, iterations = test_iterations_needed_watts(i * 10, 10, 0.1)
        needed_iterations.append(iterations)
        ns.append(G.number_of_nodes())
    sns.scatterplot(x=ns, y=needed_iterations)
    df = pd.DataFrame({'N': ns, 'needed_iter': needed_iterations})
    df.to_csv('run_times_needed_iter2.csv', index=False)
    plt.show()


def compare_infomap_gt(G):
    detector = GameTheoreticCommunityDetection(G)
    detector.run(init_method="singleton")
    im_partition = nx_infomap_partition(G)
    detector.visualize(ground_truth=im_partition, ground_truth_title="Infomap")


def plot_caveman_n_watts():
    G = nx.connected_caveman_graph(10, 10)
    set_all_edge_weights(G, 0.1, 1.0)
    compare_infomap_gt(G)
    G = nx.connected_watts_strogatz_graph(100, 10, 0.1)
    compare_infomap_gt(G)
    G = nx.barbell_graph(20, 0)
    set_all_edge_weights(G, 0.1, 1.0)
    G[19][20]['weight'] = 100
    compare_infomap_gt(G)
    plt.show()


if __name__ == "__main__":
    graph_run_time()
    graph_run_time_watts()
    graph_iterations_needed_watts()
    graph_run_time_max_iter()
    plot_caveman_n_watts()
