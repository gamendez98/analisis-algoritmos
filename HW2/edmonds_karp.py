from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import time

class EdmondsKarpGraph:
    """
    Class representing a directed graph with methods to compute the maximum flow using
    the Edmonds-Karp algorithm, a breadth-first search (BFS) implementation of the Ford-Fulkerson method.
    """
    def __init__(self, edges: list[tuple[str, str, int]]):
        """
        Initializes the graph with a list of directed edges.

        Args:
            edges (list[tuple[str, str, int]]): A list of edges in the format (node1, node2, capacity).
        """
        self.edges = edges
        self.nodes = set(n for edge in edges for n in edge[:-1])  # Extract unique nodes
        self.node_neighbours = {node: [] for node in self.nodes}  # Adjacency list representation
        for node1, node2, weight in edges:
            self.node_neighbours[node1].append((node2, weight))

        self.residual_node_neighbours: dict[str, dict[str, int]] = {}  # Residual graph representation
        self.flow_node_neighbours: dict[str, dict[str, int]] = {}  # Flow network representation
        self.init_residual()

    def init_residual(self):
        """Initializes the residual capacity graph."""
        self.residual_node_neighbours = {node: {} for node in self.nodes}
        for node1, node2, weight in self.edges:
            self.residual_node_neighbours[node1][node2] = weight

    def init_flow(self):
        """Initializes the flow network with zero flow for each edge."""
        self.flow_node_neighbours = {node: {} for node in self.nodes}
        for node1, node2, weight in self.edges:
            self.flow_node_neighbours[node1][node2] = 0

    def bfs_residual(self, source, target):
        """
        Performs BFS to find an augmenting path in the residual graph.

        Args:
            source (str): The source node.
            target (str): The target node.
        
        Returns:
            list[tuple[str, int]] | None: The augmenting path if found, otherwise None.
        """
        q = deque()
        visited = {node: False for node in self.nodes}
        visited[source] = True
        q.append(source)
        predecessors = {source: None}
        while q:
            node = q.popleft()
            for neighbour, weight in self.residual_node_neighbours[node].items():
                if weight and not visited[neighbour]:
                    visited[neighbour] = True
                    predecessors[neighbour] = (node, weight)
                    q.append(neighbour)
                if neighbour == target and weight:
                    path = [(target, None)]
                    while predecessors[path[0][0]] is not None:
                        path.insert(0, predecessors[path[0][0]])
                    return path
        return None

    def edmonds_karp(self, source, target):
        """
        Implements the Edmonds-Karp algorithm to compute the maximum flow.

        Args:
            source (str): The source node.
            target (str): The target node.
        
        Returns:
            dict[str, dict[str, int]]: The resulting flow network.
        """
        self.init_residual()
        self.init_flow()
        while True:
            path = self.bfs_residual(source, target)
            if path is None:
                return self.flow_node_neighbours # No more augmenting paths found
            self.push_flow(path)


    def push_flow(self, path):
        """
        Augments flow along the found path and updates the residual graph.

        Args:
            path (list[tuple[str, int]]): The augmenting path found by BFS.
        """
        path_flow = min(flow for _, flow in path[:-1])  # Find the bottleneck capacity
        for (node0, _), (node1, _) in zip(path, path[1:]):
            self.residual_node_neighbours[node0][node1] -= path_flow  # Update residual capacities
            self.flow_node_neighbours[node0][node1] += path_flow  # Update actual flow

    @classmethod
    def read_graph_from_file(cls, input_file) -> "EdmondsKarpGraph":
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
            lines = f.readlines()
        
        if not lines:
            raise ValueError("Input file is empty.")
        
        N = int(lines[0].strip())  # First line contains number of nodes
        edges = set()  # Use a set to check for duplicate edges

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 3:
                raise ValueError(f"Invalid line format: {line.strip()}")
            
            node1, node2, weight = map(int, parts)
            
            if (node1, node2) in edges:
                raise ValueError(f"Duplicate edge detected between {node1} and {node2}")
            
            if weight < 0:
                raise ValueError(f"Negative weight detected: {line.strip()}")

            edges.add((node1, node2, weight))

        return cls(list(edges))
    
    def write_output(self, output_file, source, target):
        """Writes the flow network result to a file."""
        start_time = time.time()
        max_flow = self.edmonds_karp(source, target)
        end_time = time.time()
        
        with open(output_file, "w") as f:
            f.write("Edge Flows:\n")
            for node1 in sorted(max_flow):
                for node2, flow in max_flow[node1].items():
                    f.write(f"{node1} {node2} {flow}\n")
            
            total_flow = sum(flow for flow in max_flow[source].values())
            f.write(f"\nTotal Flow: {total_flow}\n")
            f.write(f"Execution Time: {end_time - start_time:.6f} seconds\n")

    def plot_graph_with_flow(self, source, target):
        """Plots the flow network using Matplotlib and NetworkX."""
        g = nx.DiGraph()
        
        # Run Edmonds-Karp algorithm
        flow_result = self.edmonds_karp(source, target)

        # Define edge attributes
        edge_labels = {}
        edge_widths = {}
        edge_colors = {}

        for node1 in flow_result:
            for node2, flow in flow_result[node1].items():
                capacity = self.residual_node_neighbours[node1].get(node2, 0) + flow  # Total capacity
                edge_labels[(node1, node2)] = f"{flow}/{capacity}"  # Flow/Capacity label
                
                # Set edge width and color
                edge_widths[(node1, node2)] = 1 + 3 * flow / max(1, capacity)  # Thicker edges for more flow
                edge_colors[(node1, node2)] = "blue" if flow > 0 else "gray"  # Blue for flow, Gray for no flow

        for node1, node2, capacity in self.edges:
            g.add_edge(node1, node2, capacity=capacity) 

        # Define node positions
        pos = nx.spring_layout(g, seed=42)  

        # Get edges in NetworkX order
        edges = list(g.edges())

        # Assign colors and widths in the correct order
        edge_color_list = [edge_colors.get(edge, "gray") for edge in edges]
        edge_width_list = [edge_widths.get(edge, 1) for edge in edges]

        # Draw nodes with source and target highlighted
        node_colors = ["green" if node == source else "red" if node == target else "lightblue" for node in g.nodes]
        nx.draw(g, pos, with_labels=True, node_color=node_colors, edge_color=edge_color_list, width=edge_width_list, node_size=2000, font_size=10)

        # Draw edge labels for flow/capacity
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=9)

        # Set title
        plt.title("Flow Network after Edmonds-Karp Execution")
        plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python edmonds_karp.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    graph = EdmondsKarpGraph.read_graph_from_file(input_file)
    source = 0  # First node is source
    target = max(graph.nodes)  # Last node is target
    
    graph.write_output(output_file, source, target)
    graph.plot_graph_with_flow(source, target)
    print(f"Results written to {output_file}")