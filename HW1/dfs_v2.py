import sys
import matplotlib.pyplot as plt
import networkx as nx

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
        self.adj_list = {node: [] for node in self.nodes}  # Adjacency list representation
        for node1, node2, weight in edges:
            self.adj_list[node1].append((node2, weight))

    def plot_graph(self):
        """
        Plots the graph using matplotlib and networkx.
        """
        g = nx.DiGraph()
        for node1, node2, weight in self.edges:
            g.add_edge(node1, node2, weight=weight)

        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        edge_labels = nx.get_edge_attributes(g, 'weight')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        plt.title("Graph Visualization")
        plt.show()


def dfs(graph):
    """
    Performs a Depth-First Search (DFS) on the graph to find cycles.

    Args:
        graph (Graph): The graph for which to find cycles.

    Returns:
        list[list[str]]: List of cycles if the graph is not acyclic. If it is, it returns an empty list.
    """
    visited = {node: False for node in graph.nodes}
    recursive_stack = {node: False for node in graph.nodes}
    cycles = []
    unique_cycles = set()

    def dfs_visit(node, path):
        """
        Helper function to visit nodes during DFS.

        Args:
            node (str): The current node being visited.
            path (list[str]): The current path of nodes being explored.
        """
        visited[node] = True
        recursive_stack[node] = True
        path.append(node)

        for neighbor, _ in graph.adj_list[node]:
            if not visited[neighbor]:
                dfs_visit(neighbor, path)
            elif recursive_stack[neighbor]:
                cycle_start_index = path.index(neighbor)
                cycle = path[cycle_start_index:] # + [neighbor]  # Only to show the cycle in the output
                cycle_tuple = tuple(cycle)
                if cycle_tuple not in unique_cycles:
                    unique_cycles.add(cycle_tuple)
                    cycles.append(cycle)

        path.pop()
        recursive_stack[node] = False

    for node in graph.nodes:
        if not visited[node]:
            dfs_visit(node, [])

    return cycles


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
    edges = []
    with open(input_file, 'r') as f:
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


def write_cycles_to_file(output_file, cycles):
    """
    Writes the cycles to a .out file.

    Args:
        output_file (str): The path to the output file.
        cycles (list[list[str]]): The cycles to write.
    """
    with open(output_file, 'w') as f:
        if len(cycles) == 0:
            f.write("No self-loans were detected\n")
        else:
            for cycle in cycles:
                f.write(" -> ".join(cycle) + "\n")  # Write each cycle in the format "node1 -> node2 -> ... -> nodek\n"


def main():
    """
    Main function to read input, execute DFS algorithm, and write output.
    Accepts input and output file names as command-line arguments.
    """
    if len(sys.argv) != 3:  # Ensure correct number of arguments
        print("Usage: python dfs.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        graph = read_graph_from_file(input_file)  # Read the graph from the input file
        graph.plot_graph()  # Plot the graph
        cycles = dfs(graph)  # Compute cycles using DFS
        write_cycles_to_file(output_file, cycles)  # Write the cycles to the output file
    except ValueError as e:
        print(f"Error: {e}")  # Print error message if an exception occurs
        sys.exit(1)


if __name__ == '__main__':
    main()
