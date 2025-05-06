import networkx as nx
from networkx import Graph

from infomap import Infomap
from tqdm import tqdm
import random
import time

#%%

def nx_infomap_partition(G : Graph):
    """
    Partitions the nodes of a graph using the Infomap algorithm.

    This function applies the Infomap community detection algorithm to partition the input
    graph into communities. The algorithm considers the edges and their associated weights
    (if any) in the graph to determine the communities.

    Args:
        G: A NetworkX graph. Can be either directed or undirected. If edge weights are
            not specified, a default weight of 1 will be used.

    Returns:
        dict: A dictionary where each key is a node ID from the graph, and its corresponding
            value is the ID of the community (module) to which the node belongs.
    """
    infomap = Infomap(
        "--two-level --directed"
    )
    for edge in G.edges:
        n, m = edge
        w = G.edges[edge].get('weight', 1)
        infomap.add_link(n, m, w)
    infomap.run(silent=True)
    partition = {}
    for node in infomap.tree:
        if node.is_leaf:
            partition[node.node_id]=node.module_id

    return partition

def set_all_edge_weights(G, low, high):
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(low, high)


def test_run_time_watts_infomap(n, k, p, max_iter=100):
    G = nx.connected_watts_strogatz_graph(n, k, p)
    set_all_edge_weights(G, 0.1, 1.0)
    start_time = time.time()
    _ = nx_infomap_partition(G)
    end_time = time.time()
    run_time = end_time - start_time
    return G, run_time


def graph_run_time_watts_infomap():
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    run_times = []
    ns = []
    es = []
    for i in tqdm(range(1, 1000)):
        G, run_time = test_run_time_watts_infomap(i * 10, 10, 0.1)
        run_times.append(run_time)
        n = G.number_of_nodes()
        e = G.number_of_edges()
        es.append(e)
        ns.append(n)
    sns.scatterplot(x=ns, y=run_times)
    df = pd.DataFrame({'n': ns, 'e': es, 'run_time': run_times})
    df.to_csv('run_times_watts2_infomap.csv', index=False)
    plt.show()


if __name__ == "__main__":
    graph_run_time_watts_infomap()