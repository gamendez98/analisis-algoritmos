from collections import deque

import networkx as nx
import matplotlib.pyplot as plt


class EdmondsKarpGraph:
    def __init__(self, edges: list[tuple[str, str, int]]):
        """
        Initializes the Graph instance with edges and nodes.

        Args:
            edges (list[tuple[str, str, int]]): List of edges in the format (node1, node2, weight).
        """
        self.edges = edges
        # Extracts all unique nodes from the edges list
        self.nodes = set(n for edge in edges for n in edge[:-1])
        self.node_neighbours = {node: [] for node in self.nodes}  # Adjacency list representation
        for node1, node2, weight in edges:
            self.node_neighbours[node1].append((node2, weight))

        self.residual_node_neighbours: dict[str, dict[str, int]] = {}
        self.flow_node_neighbours: dict[str, dict[str, int]] = {}
        self.init_residual()

    def init_residual(self):
        self.residual_node_neighbours = {node: {} for node in self.nodes}
        for node1, node2, weight in self.edges:
            self.residual_node_neighbours[node1][node2] = weight

    def init_flow(self):
        self.flow_node_neighbours = {node: {} for node in self.nodes}
        for node1, node2, weight in self.edges:
            self.flow_node_neighbours[node1][node2] = 0

    def bfs_residual(self, source, target):
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
        self.init_residual()
        self.init_flow()
        while True:
            path = self.bfs_residual(source, target)
            if path is None:
                return self.flow_node_neighbours
            self.push_flow(path)


    def push_flow(self, path):
        path_flow = min(flow for _, flow in path[:-1])
        for (node0, weight), (node1, _) in zip(path, path[1:]):
            self.residual_node_neighbours[node0][node1] -= path_flow
            # self.residual_node_neighbours[node1][node0] += path_flow
            self.flow_node_neighbours[node0][node1] += path_flow

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
        return cls(edges)


# %%
if __name__ == "__main__":
    g = EdmondsKarpGraph.read_graph_from_file("HW2/edmonds_karp.in")
    g.plot_graph()
    print(g.edmonds_karp("A", "F"))
    # print(g.bfs_residual("A", "F"))
