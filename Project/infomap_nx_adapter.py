from networkx import Graph

from infomap import Infomap


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
    infomap.run()

    infomap.run()

    partition = {}
    for node in infomap.tree:
        if node.is_leaf:
            partition[node.node_id]=node.module_id

    return partition

