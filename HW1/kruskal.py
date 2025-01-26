import sys

class Graph:
    """
    A class to represent a graph using edges and nodes.

    Attributes:
        edges (list[tuple[str, str, int]]): A list of edges in the format (node1, node2, weight).
        nodes (set[str]): A set of all unique nodes in the graph.
    """
    def __init__(self, edges):
        """
        Initializes the Graph instance with edges and nodes.

        Args:
            edges (list[tuple[str, str, int]]): List of edges in the format (node1, node2, weight).
        """
        self.edges = edges
        # Extracts all unique nodes from the edges list
        self.nodes = set(n for edge in edges for n in edge[:-1])

def join_partial_trees(partial_trees, node0, node1):
    """
    Joins two partial trees by merging their node sets.

    Args:
        partial_trees (dict[str, set]): A dictionary mapping nodes to their respective sets.
        node0 (str): A node from the first tree.
        node1 (str): A node from the second tree.
    """
    resulting_tree_nodes = partial_trees[node0]  # Get the set of nodes for the first tree
    merged_tree_nodes = partial_trees[node1]  # Get the set of nodes for the second tree
    resulting_tree_nodes.update(merged_tree_nodes)  # Merge the two sets
    for node in merged_tree_nodes:
        partial_trees[node] = resulting_tree_nodes  # Update all nodes in the merged set

def kruskal(graph):
    """
    Implements Kruskal's algorithm to find the Minimum Spanning Tree (MST) of a graph.

    Args:
        graph (Graph): The graph for which to find the MST.

    Returns:
        list[tuple[str, str, int]]: A list of edges in the MST.
    """
    n_nodes = len(graph.nodes)  # Number of nodes in the graph
    selected_edges = 0  # Count of edges added to the MST
    edges = sorted(graph.edges, key=lambda x: x[2])  # Sort edges by weight in ascending order
    partial_trees_nodes = {node: {node} for node in graph.nodes}  # Each node is its own tree initially
    tree_edges = []  # List to store the edges of the MST

    for node1, node2, weight in edges:
        # Check if the nodes belong to different trees
        if partial_trees_nodes[node1] != partial_trees_nodes[node2]:
            tree_edges.append((node1, node2, weight))  # Add the edge to the MST
            join_partial_trees(partial_trees_nodes, node1, node2)  # Merge the trees
            selected_edges += 1
            # Stop if we have n-1 edges (MST is complete)
            if selected_edges == n_nodes - 1:
                break

    # Ensure all nodes are connected; otherwise, the graph is disconnected
    root_tree = partial_trees_nodes[next(iter(graph.nodes))]
    if len(root_tree) != n_nodes:
        raise ValueError("The graph is disconnected, and no MST spans all nodes.")

    return tree_edges

def read_graph_from_file(input_file):
    """
    Reads graph edges from a .in file.

    Args:
        input_file (str): The path to the input file.

    Returns:
        Graph: A graph instance constructed from the file.

    Raises:
        ValueError: If any edge has a negative weight or if the file format is invalid.
    """
    with open(input_file, 'r') as f:
        edges = []
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:  # Validate line format
                raise ValueError(f"Invalid line format: {line.strip()}")
            node1, node2, weight = parts
            weight = int(weight)  # Convert weight to integer
            if weight < 0:  # Check for negative weights
                raise ValueError(f"Negative weight detected: {line.strip()}")
            edges.append((node1, node2, weight))  # Add the edge to the list
    return Graph(edges)

def write_mst_to_file(output_file, mst):
    """
    Writes the MST edges to a .out file.

    Args:
        output_file (str): The path to the output file.
        mst (list[tuple[str, str, int]]): The MST edges to write.
    """
    with open(output_file, 'w') as f:
        for edge in mst:
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")  # Write each edge in the format "node1 node2 weight"

def main():
    """
    Main function to read input, execute Kruskal's algorithm, and write output.
    Accepts input and output file names as command-line arguments.
    """
    if len(sys.argv) != 3:  # Ensure correct number of arguments
        print("Usage: python kruskal.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        graph = read_graph_from_file(input_file)  # Read the graph from the input file
        mst = kruskal(graph)  # Compute the MST using Kruskal's algorithm
        write_mst_to_file(output_file, mst)  # Write the MST to the output file
    except ValueError as e:
        print(f"Error: {e}")  # Print error message if an exception occurs
        sys.exit(1)

if __name__ == '__main__':
    main()