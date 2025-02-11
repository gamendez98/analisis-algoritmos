import networkx as nx
import matplotlib.pyplot as plt
import time

class PushRelabel:
    """
    Class representing a directed graph with methods to compute the maximum flow using
    the the push-relabel algorithm.
    """
    def __init__(self, edges: list[tuple[str, str, int]]):
        """
        Initializes the graph with a list of directed edges.

        Args:
            edges (list[tuple[str, str, int]]): A list of edges in the format (node1, node2, capacity).
        """
        self.edges = edges
        self.nodes = set(n for edge in edges for n in edge[:-1])
        self.node_neighbours = {node: [] for node in self.nodes}
        for node1, node2, weight in edges:
            self.node_neighbours[node1].append((node2, weight))

        self.residual_node_neighbours = {}
        self.flow_node_neighbours = {}
        self.init_residual()
        self.init_flow()

    def init_residual(self):
        """Initializes the residual capacity graph."""
        self.residual_node_neighbours = {node: {} for node in self.nodes}
        for node1, node2, weight in self.edges:
            self.residual_node_neighbours[node1][node2] = weight
            self.residual_node_neighbours[node2][node1] = 0

    def init_flow(self):
        """Initializes the flow network with zero flow for each edge."""
        self.flow_node_neighbours = {node: {} for node in self.nodes}
        for node1, node2, weight in self.edges:
            self.flow_node_neighbours[node1][node2] = 0
            self.flow_node_neighbours[node2][node1] = 0

    def init_preflow(self, source):
        """Initializes preflow from source and sets initial height."""
        self.height = {node: 0 for node in self.nodes}
        self.height[source] = len(self.nodes)
        
        self.excess = {node: 0 for node in self.nodes}
        self.active_nodes = set()
        
        # Push flow from source to all neighbors
        for v in self.residual_node_neighbours[source]:
            capacity = self.residual_node_neighbours[source][v]
            if capacity > 0:

                self.flow_node_neighbours[source][v] = capacity
                self.flow_node_neighbours[v][source] = 0

                self.residual_node_neighbours[source][v] = 0
                self.residual_node_neighbours[v][source] = capacity
                
                self.excess[v] = capacity
                if v != source:
                    self.active_nodes.add(v)

    def push(self, u, v):
        """Pushes flow from node u to node v."""
        flow = min(self.excess[u], self.residual_node_neighbours[u][v])
        if flow > 0:
            self.flow_node_neighbours[u][v] = self.flow_node_neighbours.get(u, {}).get(v, 0) + flow
            self.flow_node_neighbours[v][u] = self.flow_node_neighbours.get(v, {}).get(u, 0) - flow
            self.residual_node_neighbours[u][v] -= flow
            self.residual_node_neighbours[v][u] += flow
            self.excess[u] -= flow
            self.excess[v] += flow
            return True
        return False

    def relabel(self, u):
        """Relabels the height of node u."""
        min_height = float("inf")
        for v in self.residual_node_neighbours[u]:
            if self.residual_node_neighbours[u][v] > 0:
                min_height = min(min_height, self.height[v])
        if min_height != float("inf"):
            self.height[u] = min_height + 1
            return True
        return False

    def push_relabel(self, source, target):
        """Computes the maximum flow from source to target."""
        self.init_preflow(source)
        
        while self.active_nodes:
            u = next(iter(self.active_nodes))
            pushed = False
            for v in self.residual_node_neighbours[u]:
                if (self.residual_node_neighbours[u][v] > 0 and 
                    self.height[u] == self.height[v] + 1):
                    if self.push(u, v):
                        pushed = True
                        if v != source and v != target and self.excess[v] > 0:
                            self.active_nodes.add(v)
                        if self.excess[u] == 0:
                            self.active_nodes.remove(u)
                            break
            
            if not pushed:
                if self.relabel(u):
                    if self.excess[u] > 0:
                        continue
                self.active_nodes.remove(u)
        
        return sum(self.flow_node_neighbours[source].values()) 

    def plot_graph_with_flow(self, source, target):
        """
        Plots the flow network after running the Push-Relabel algorithm.
        Uses dictionaries for edge attributes to ensure correct mapping.
        """
        g = nx.DiGraph()
        
        # Define edge attributes using dictionaries
        edge_labels = {}
        edge_widths = {}
        edge_colors = {}
        
        # Process all edges and set their attributes
        for node1 in self.flow_node_neighbours:
            for node2, flow in self.flow_node_neighbours[node1].items():
                # Get the capacity from original edges
                capacity = 0
                for n1, n2, cap in self.edges:
                    if n1 == node1 and n2 == node2:
                        capacity = cap
                        break
                
                if capacity > 0:  # Only add edges that existed in original graph
                    edge_labels[(node1, node2)] = f"{flow}/{capacity}"
                    edge_widths[(node1, node2)] = 1 + 3 * flow / max(1, capacity)
                    edge_colors[(node1, node2)] = "blue" if flow > 0 else "gray"
        
        # Add edges to NetworkX graph
        for node1, node2, capacity in self.edges:
            g.add_edge(node1, node2, capacity=capacity)
        
        # Define node positions using Kamada-Kawai layout
        pos = nx.kamada_kawai_layout(g)
        
        # Get edges in NetworkX order
        edges = list(g.edges())
        
        # Map attributes to edges in correct order
        edge_color_list = [edge_colors.get(edge, "gray") for edge in edges]
        edge_width_list = [edge_widths.get(edge, 1) for edge in edges]
        
        # Draw nodes with source and target highlighted
        node_colors = ["green" if node == source 
                    else "red" if node == target 
                    else "lightblue" for node in g.nodes()]
        
        # Draw the graph
        nx.draw(g, pos,
                with_labels=True,
                node_color=node_colors,
                edge_color=edge_color_list,
                width=edge_width_list,
                node_size=2000,
                font_size=10,
                arrows=True,
                arrowsize=20)
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=9)
        
        plt.title("Flow Network")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @classmethod
    def read_graph_from_file(cls, input_file):
        """Reads graph from input file."""
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError("Input file is empty.")
        
        edges = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            node1, node2, weight = map(int, parts)
            edges.append((str(node1), str(node2), int(weight)))
        
        return cls(edges)

    def write_output(self, output_file, source, target):
        """Writes results to output file."""
        start_time = time.time()
        max_flow = self.push_relabel(source, target)
        end_time = time.time()
        
        with open(output_file, "w") as f:
            f.write("Edge Flows:\n")
            for node1 in sorted(self.flow_node_neighbours):
                for node2 in sorted(self.flow_node_neighbours[node1]):
                    flow = self.flow_node_neighbours[node1][node2]
                    if flow > 0:
                        f.write(f"{node1} {node2} {flow}\n")
            f.write(f"\nMaximum Flow: {max_flow}\n")
            f.write(f"Execution Time: {end_time - start_time:.6f} seconds\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python push_relabel.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    graph = PushRelabel.read_graph_from_file(input_file)
    source = "0"
    target = str(max(int(node) for node in graph.nodes))
    
    graph.write_output(output_file, source, target)
    graph.plot_graph_with_flow(source, target)
    print(f"Results written to {output_file}")