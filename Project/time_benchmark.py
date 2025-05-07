import concurrent
import os
import pickle
import sys

from Project.infomap_impl import Infomap

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

def create_graphs(n_samples, graph_sizes, output_dir):
    graphs = []
    random_seed = 1
    sizes = []
    for samples, size in zip(n_samples, graph_sizes):
        sizes.extend([size] * samples)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for graph_size in tqdm(sizes):
            min_community = 20
            def f_call():
                return LFR_benchmark_graph(
                    n=graph_size,
                    tau1=2.0,
                    tau2=1.5,
                    mu=0.5,  # More flexibility
                    average_degree=10,
                    min_community=min_community,
                    seed=random_seed,
                )
            while True:
                random_seed += 1
                future = executor.submit(f_call)
                try:
                    result = future.result(timeout=20)  # 5-second timeout
                    with open(f"{output_dir}/benchmark_graph_{graph_size}_{random_seed}.pkl", "wb") as f:
                        pickle.dump(result, f)
                    break
                except concurrent.futures.TimeoutError:
                    print("Function timed out.")
                except networkx.exception.ExceededMaxIterations as e:
                    print(f"Failed to generate graph of size with {type(e)} {e}")
                    min_community += min(min_community, graph_size)
                except Exception as e:
                    print(f"Failed to generate graph of size with {type(e)} {e}")

    return graphs


def main():
    info = []
    get_gt_partition = lambda test_graph: GameTheoreticCommunityDetection(test_graph).run()[0]
    get_native_infomap_partition = lambda test_graph: Infomap(nx.DiGraph(test_graph)).run()[0]
    for graph_file in tqdm(os.listdir('test_graphs')):
        with open(f'test_graphs/{graph_file}', 'rb') as f:
            graph = pickle.load(f)
        graph_size = graph.number_of_nodes()
        print(f"Running Game Theory for graph of size {graph_size}... ", end="")
        gt_info = calculate_metrics(graph, get_gt_partition)
        gt_info.update({'nodes': graph_size, 'algorithm': 'game_theory'})
        print("Done!")
        print(f"Running Infomap for graph of size {graph_size}... ", end="")
        infomap_info = calculate_metrics(graph, nx_infomap_partition)
        infomap_info.update({'nodes': graph_size, 'algorithm': 'infomap'})
        print("Done!")
        print(f"Running Infomap (Native) for graph of size {graph_size}... ", end="")
        infomap_native_info = calculate_metrics(graph, get_native_infomap_partition)
        infomap_native_info.update({'nodes': graph_size, 'algorithm': 'infomap_native'})
        print("Done!")

        info.append(gt_info)
        info.append(infomap_info)
        info.append(infomap_native_info)

    df = pd.DataFrame(info)
    df.to_csv('benchmark_results_3.csv', index=False)
    print(df.groupby(['algorithm', 'nodes']).describe())


if __name__ == "__main__":
    # create_graphs([10, 10, 10], [250, 500, 1000])
    main()
