from collections import deque
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
        
        edges = set() 

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
    
    def structure_output(self, source, target):
        """
        Structures the flow network result into a readable string.

        Args:
            source (str): The source node.
            target (str): The target node.

        Returns:
            str: The formatted output with edge flows and execution time.
        """
        start_time = time.time()
        max_flow = self.edmonds_karp(source, target)
        end_time = time.time()

        s = "Edge Flows:\n"
        for node1 in sorted(max_flow):
            for node2, flow in max_flow[node1].items():
                s += f"{node1} {node2} {flow}\n"
        
        total_flow = sum(flow for flow in max_flow[source].values())
        s += f"\nTotal Flow: {total_flow}\n"
        s += f"Execution Time: {end_time - start_time} seconds\n"
        return s

class PushRelabelGraph:
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
        for node1, node2, _ in self.edges:
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

    @classmethod
    def read_graph_from_file(cls, input_file) -> "PushRelabelGraph":
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
        
        edges = set() 

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 3:
                raise ValueError(f"Invalid line format: {line.strip()}")
            
            node1, node2, weight = map(int, parts)
            
            if (node1, node2) in edges:
                raise ValueError(f"Duplicate edge detected between {node1} and {node2}")
            
            if weight < 0:
                raise ValueError(f"Negative weight detected: {line.strip()}")

            edges.add((str(node1), str(node2), weight))

        return cls(list(edges))
    
    def structure_output(self, source, target):
        """
        Structures the flow network result into a readable string.

        Args:
            source (str): The source node.
            target (str): The target node.

        Returns:
            str: The formatted output with edge flows and execution time.
        """
        start_time = time.time()
        max_flow = self.push_relabel(source, target)
        end_time = time.time()
        s = "Edge Flows:\n"
        for node1 in sorted(self.flow_node_neighbours):
            for node2 in sorted(self.flow_node_neighbours[node1]):
                flow = self.flow_node_neighbours[node1][node2]
                if flow > 0:
                    s += f"{node1} {node2} {flow}\n"
        s += f"\nMaximum Flow: {max_flow}\n"
        s += f"Execution Time: {end_time - start_time:.6f} seconds\n"
        return s

def run_algorithms(input_file, output_file):
    """
    Runs the Edmonds-Karp and Push-Relabel algorithms on the input file 
    and writes the results to the output file.

    Args:
        input_file (str): Path to the input file containing the graph.
        output_file (str): Path to the output file where results will be written.
    """
    graph_edmonds_karp = EdmondsKarpGraph.read_graph_from_file(input_file)
    graph_push_relabel = PushRelabelGraph.read_graph_from_file(input_file)
    
    ek_output = graph_edmonds_karp.structure_output(0, max(int(node) for node in graph_edmonds_karp.nodes))
    pr_output = graph_push_relabel.structure_output("0", str(max(int(node) for node in graph_push_relabel.nodes)))
    
    with open(output_file, "w") as f:
        f.write("Algorithm Results:\n\n")
        f.write("Edmonds-Karp Algorithm\n")
        f.write(ek_output)
        f.write("\n")
        f.write("Push-Relabel Algorithm\n")
        f.write(pr_output)
    
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python maximum_flow.py <input_file> <output_file>")
        sys.exit(1)
    run_algorithms(sys.argv[1], sys.argv[2])