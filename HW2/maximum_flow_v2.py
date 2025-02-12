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
    
class Edge:

    def __init__(self, flow, capacity, u, v):
        self.flow = flow
        self.capacity = capacity
        self.u = u
        self.v = v

class Vertex:

    def __init__(self, h, e):
        self.h = h
        self.e = e

class PushRelabelGraph:
    """
    Class representing a directed graph with methods to compute the maximum flow using
    the the push-relabel algorithm.
    """
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = []
        self.vertices = []

        for i in range(num_vertices):
            #Each vertex starts with height of 0 and excess flow of 0
            self.vertices.append(Vertex(0, 0))
    
    def addEdge(self, u, v, capacity):
        #flow is initialized with 0 for all edges
        self.edges.append(Edge(0, capacity, u, v))

    def nextFlowVertex(self):
        #Might create a data structrure that only holds the excess nodes

        for i in range(1, self.num_vertices - 1):
            if self.vertices[i].e > 0:
                return i
            
        return -1

    def init_preflow(self, s):

        self.vertices[s].h = len(self.vertices)

        for i  in range(len(self.edges)):

            if self.edges[i].u == s:

                self.edges[i].flow = self.edges[i].capacity

                self.vertices[self.edges[i].v].e += self.edges[i].flow

                self.edges.append(Edge(-self.edges[i].flow, 0, self.edges[i].v, s))

    def updateReverseFlow(self, i, flow):

        initial = self.edges[i].v
        final = self.edges[i].u
        found = False
        j = 0
        while j < len(self.edges) and found == False:  
            if (self.edges[j].v == final and self.edges[j].u == initial):
                self.edges[j].flow -= flow
                found = True
            j +=1
        if found == False:
            # adding reverse Edge in residual graph 
            e = Edge(0, flow, initial, final)
            self.edges.append(e)
    
    def push(self, u):

        for i in range(len(self.edges)):

            if self.edges[i].u == u:

                if self.edges[i].flow == self.edges[i].capacity:
                    continue

                if self.vertices[u].h == self.vertices[self.edges[i].v].h + 1:

                    flow = min(self.vertices[u].e, self.edges[i].capacity - self.edges[i].flow)

                    self.vertices[u].e -= flow

                    self.vertices[self.edges[i].v].e += flow

                    self.edges[i].flow += flow

                    self.updateReverseFlow(i, flow)

                    return True
        return False
    
    def relabel(self, u):
        
        min_height = float('inf')

        for i in range(len(self.edges)):

            if self.edges[i].u == u:

                if self.edges[i].flow == self.edges[i].capacity:
                    continue

                if self.vertices[self.edges[i].v].h < min_height:
                    min_height = self.vertices[self.edges[i].v].h

        self.vertices[u].h = min_height + 1
    
    def push_relabel(self, s, t):

        self.init_preflow(s)

        u = self.nextFlowVertex()

        while u != -1:

            if not self.push(u):
                self.relabel(u)

            u = self.nextFlowVertex()

        return self.vertices[t].e
    
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
    
def read_graph_from_file_push_relabel(input_file):
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
        vertices = int(lines[0])

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

            

        return list(edges), vertices

def run_algorithms(input_file, output_file):
    """
    Runs the Edmonds-Karp and Push-Relabel algorithms on the input file 
    and writes the results to the output file.

    Args:
        input_file (str): Path to the input file containing the graph.
        output_file (str): Path to the output file where results will be written.
    """

    #Edmonds_karp
    graph_edmonds_karp = EdmondsKarpGraph.read_graph_from_file(input_file)  
    ek_output = graph_edmonds_karp.structure_output(0, max(int(node) for node in graph_edmonds_karp.nodes))
    
    #Push-Relabel
    edges, V = read_graph_from_file_push_relabel(input_file)
    graph_push_relabel = PushRelabelGraph(int(V))
    print(V)
    for edge in edges:
        graph_push_relabel.addEdge(edge[0], edge[1], edge[2])
    push_relable_max_flow = graph_push_relabel.push_relabel(0, int(V-1))
    #pr_output = graph_push_relabel.structure_output("0", str(max(int(node) for node in graph_push_relabel.nodes)))
    
    with open(output_file, "w") as f:
        f.write("Algorithm Results:\n\n")
        f.write("Edmonds-Karp Algorithm\n")
        f.write(ek_output)
        f.write("\n")
        f.write("Push-Relabel Algorithm\n")
        f.write(str(push_relable_max_flow))
    
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python maximum_flow.py <input_file> <output_file>")
        sys.exit(1)
    run_algorithms(sys.argv[1], sys.argv[2])