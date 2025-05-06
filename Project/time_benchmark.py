import sys
sys.path.append('/home/gustavo/Documents/analisis-algoritmos/Project')
from itertools import product

import networkx
import networkx as nx
import pandas as pd
from networkx import LFR_benchmark_graph

from game_theory import GameTheoreticCommunityDetection
from infomap_nx_adapter import nx_infomap_partition
import time
from tqdm import tqdm



def community_from_g_lrf(glrf):
    communities = []
    for node in glrf.nodes().values():
        community = node['community']
        if community not in communities:
            communities.append(community)
    return {
        node_id: community_id
        for community_id, community in enumerate(communities)
        for node_id in community
    }


def time_it(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def calculate_metrics(graph, partition_function):
    selected_metrics = ['modularity', 'ari', 'nmi']
    dummy = GameTheoreticCommunityDetection(graph)
    partition, run_time = time_it(partition_function, graph)
    dummy.partition = partition
    ground_truth = community_from_g_lrf(graph)
    metrics = dummy.calculate_community_metrics(ground_truth=ground_truth)
    s_metrics = {}
    for select in selected_metrics:
        s_metrics[select] = metrics[select]
    s_metrics['time'] = run_time
    return s_metrics


def get_LFR_benchmark_graph(size):
    min_community = 20
    while True:
        try:
            g = LFR_benchmark_graph(
                n=size,
                tau1=2.0,
                tau2=1.5,
                mu=0.5,  # More flexibility
                average_degree=10,
                min_community=min_community
            )
            return g
        except networkx.exception.ExceededMaxIterations as e:
            print(f"Failed to generate graph of size with {type(e)} {e}")
            min_community += min(min_community, size)
        except Exception as e:
            print(f"Failed to generate graph of size with {type(e)} {e}")
    return None


# %%

def main():
    n_samples = 15
    graph_sizes = [500, 1000, 2000]
    info = []
    get_gt_partition = lambda test_graph: GameTheoreticCommunityDetection(test_graph).run()[0]
    for _, graph_size in tqdm(product(range(n_samples), graph_sizes), total=n_samples * len(graph_sizes)):
        graph = get_LFR_benchmark_graph(graph_size)
        gt_info = calculate_metrics(graph, get_gt_partition)
        gt_info.update({'nodes': graph_size, 'algorithm': 'game_theory'})
        infomap_info = calculate_metrics(graph, nx_infomap_partition)
        infomap_info.update({'nodes': graph_size, 'algorithm': 'infomap'})
        info.append(gt_info)
        info.append(infomap_info)

    df = pd.DataFrame(info)
    df.to_csv('benchmark_results_2.csv', index=False)
    print(df.groupby(['algorithm', 'nodes']).describe())


if __name__ == "__main__":
    main()
