#from collections import deque
#import networkx as nx
#import matplotlib.pyplot as plt
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
        self.nodes = set(n for edge in edges for n in edge[:-1])  # Extract unique nodes
        self.node_neighbours = {node: [] for node in self.nodes}  # Adjacency list representation
        for node1, node2, weight in edges:
            self.node_neighbours[node1].append((node2, weight))

        self.residual_node_neighbours: dict[str, dict[str, int]] = {}  # Residual graph representation
        self.flow_node_neighbours: dict[str, dict[str, int]] = {}  # Flow network representation
        self.init_residual()
        self.init_flow()
        self.init_height(0)
        self.init_excess(0)

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

    def init_height(self, source):
        """Initializes the height of each node."""
        self.height = {node: 0 for node in self.nodes}
        self.height[source] = len(self.nodes)

    def init_excess(self, source):
        """Initializes the excess flow at each node."""
        self.excess = {node: 0 for node in self.nodes}
        self.active_nodes = {}
        for v in self.residual_node_neighbours[source]:
            #Get the capacity of the edge
            capacity = self.residual_node_neighbours[source][v]

            #Set the flow of the edge to the max capacity
            self.flow_node_neighbours[source][v] = capacity
            self.flow_node_neighbours[v][source] = -capacity

            #Set the reverse edge to the max capacity   
            self.residual_node_neighbours[source][v] = 0 #Edge is saturated
            self.residual_node_neighbours[v][source] = capacity #The first reverse edge we have

            #Set the excess of the neighbor node of the source to the capacity
            self.excess[v] = capacity
            self.active_nodes[v] = True

        self.excess[source] = -sum(self.excess.values())

    def push(self, u, v):
        """Pushes flow from node u to node v."""
        u_excess = self.excess[u]
        capacity = self.residual_node_neighbours[u][v]

        flow = min(u_excess, capacity)

        #Update flow graph
        self.flow_node_neighbours[u][v] += flow
        self.flow_node_neighbours[v][u] -=  flow
   

        #Update residual graph
        self.residual_node_neighbours[u][v] -= flow
        self.residual_node_neighbours[v][u] += flow

        #Update excess
        self.excess[u] -= flow
        self.excess[v] += flow
        
    def relabel(self, u):
        """Relabels the height of node u."""
        min_height = float("inf")
        for v in self.residual_node_neighbours[u]:
            if self.residual_node_neighbours[u][v] > 0:
                min_height = min(min_height, self.height[v])
        self.height[u] = min_height + 1

    def push_relabel(self, source, target):
        """Computes the maximum flow from the source to the target node."""
        
        while len(self.active_nodes) > 0:
            u = self.active_nodes.popitem()[0]
            pushed = False
            for v in self.residual_node_neighbours[u]:
                if self.residual_node_neighbours[u][v] > 0 and self.height[u] == self.height[v] + 1:
                    self.push(u, v)
                    pushed = True
                    if v != source and v != target and self.excess[v] > 0 and v not in self.active_nodes:
                        self.active_nodes[v] = True
                    if self.excess[u] == 0:
                        break
            if not pushed:
                self.relabel(u)
                self.active_nodes[u] = True
        return sum(self.flow_node_neighbours[v][target] for v in self.flow_node_neighbours if target in self.flow_node_neighbours[v])


    @classmethod
    def read_graph_from_file(cls, input_file) -> "PushRelabel":
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
        max_flow = self.push_relabel(source, target)
        end_time = time.time()
        
        with open(output_file, "w") as f:
            '''f.write("Edge Flows:\n")
            for node1 in sorted(max_flow):
                for node2, flow in max_flow[node1].items():
                    f.write(f"{node1} {node2} {flow}\n")
            
            total_flow = sum(flow for flow in max_flow[source].values())'''
            f.write(f"\nTotal Flow: {max_flow}\n")
            f.write(f"Execution Time: {end_time - start_time:.6f} seconds\n")

    


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python push_relabel.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    graph = PushRelabel.read_graph_from_file(input_file)
    source = 0  # First node is source
    target = max(graph.nodes)  # Last node is target
    
    graph.write_output(output_file, source, target)
    print(f"Results written to {output_file}")