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
        """Initializes the flow network with zero flow for each edge, including reverse edges."""
        self.flow_node_neighbours = {node: {} for node in self.nodes}
        for node1, node2, _ in self.edges:
            self.flow_node_neighbours[node1][node2] = 0
            
            if node2 not in self.flow_node_neighbours:
                self.flow_node_neighbours[node2] = {}
            if node1 not in self.flow_node_neighbours[node2]:
                self.flow_node_neighbours[node2][node1] = 0

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
            # Update forward flow
            self.residual_node_neighbours[node0][node1] -= path_flow
            self.flow_node_neighbours[node0][node1] += path_flow

            # Update reverse flow
            if node1 not in self.residual_node_neighbours:
                self.residual_node_neighbours[node1] = {}
            if node0 not in self.residual_node_neighbours[node1]:
                self.residual_node_neighbours[node1][node0] = 0 

            self.residual_node_neighbours[node1][node0] += path_flow


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
                if flow > 0:
                    s += f"{node1} {node2} {flow}\n"
        
        total_flow = sum(flow for flow in max_flow[source].values())
        s += f"\nTotal Flow: {total_flow}\n"
        s += f"Execution Time: {end_time - start_time} seconds\n"
        return s
    
class Edge:
    """
    Class representing an edge in a flow network.
    As per the problem, each edge has a flow, capacity, and vertices u (origin vertex) and v (destiny vertex).
    """

    def __init__(self, flow, capacity, u, v):
        self.flow = flow
        self.capacity = capacity
        self.u = u
        self.v = v

class Vertex:
    """
    Class representing a vertex in a flow network.
    As per the problem, each vertex has a height h and excess flow e.
    """

    def __init__(self, h, e):
        self.h = h
        self.e = e

class PushRelabelGraph:
    """
    Class representing a directed graph with methods to compute the maximum flow using
    the the push-relabel algorithm.
    """
    def __init__(self, num_vertices):
        """
        Initializes the graph with a number of vertices.

        Args:
            num_vertices (int): The number of vertices in the graph.
        """
        self.num_vertices = num_vertices
        self.edges = []
        self.vertices = []

        for i in range(num_vertices):
            #Each vertex starts with height of 0 and excess flow of 0
            self.vertices.append(Vertex(0, 0))
    
    def addEdge(self, u, v, capacity):
        """
        Adds an edge to the flow network.

        Args:
            u (int): The origin vertex.
            v (int): The destiny vertex.
            capacity (int): The capacity of the edge.
        """
        #Flow is initialized with 0 for all edges
        self.edges.append(Edge(0, capacity, u, v))

    def nextFlowVertex(self):
        """
        Finds the next vertex with excess flow.
        """

        for i in range(1, self.num_vertices - 1):
            if self.vertices[i].e > 0:
                return i
            
        return -1

    def init_preflow(self, s):
        """
        Initializes the preflow with the source vertex.

        Args:
            s (int): The source vertex.
        """
        #Making h of source Vertex equal to no. of vertices
        self.vertices[s].h = len(self.vertices)

        for i  in range(len(self.edges)):
            #If current edge goes from source
            if self.edges[i].u == s:
                #Flow is equal to capacity
                self.edges[i].flow = self.edges[i].capacity
                #Initialize excess flow for adjacent v equal to the flow, equal to the capacity of the edge s->v
                self.vertices[self.edges[i].v].e += self.edges[i].flow
                #Add an edge from v to s in residual graph with capacity equal to 0 and flow equal to -flow
                self.edges.append(Edge(-self.edges[i].flow, 0, self.edges[i].v, s))

    def updateReverseFlow(self, i, flow):
        """
        Updates the reverse flow of an edge in the residual graph.
        
        Args:
            i (int): The index of the edge.
            flow (int): The flow to be subtracted from the reverse edge.
        """
        #The initial vertex is now the final vertex of the edge and vice versa
        initial = self.edges[i].v
        final = self.edges[i].u
        found = False
        j = 0
        while j < len(self.edges) and found == False:  
            #If the reverse edge already exists, subtract the flow from it
            if (self.edges[j].v == final and self.edges[j].u == initial):
                self.edges[j].flow -= flow
                found = True
            j +=1
        if found == False:
            # Adding reverse Edge in residual graph 
            e = Edge(0, flow, initial, final)
            self.edges.append(e)
    
    def push(self, u):
        """
        Pushes excess flow from a vertex to a neighboring vertex.
        
        Args:
            u (int): The vertex with excess flow.   
        """

        for i in range(len(self.edges)):
            #If the edge starts at the vertex u
            if self.edges[i].u == u:
                #If the edge is already at full capacity, continue
                if self.edges[i].flow == self.edges[i].capacity:
                    continue
                #If the height of the vertex u is equal to the height of the vertex v + 1
                if self.vertices[u].h == self.vertices[self.edges[i].v].h + 1:
                    #The flow is the minimum between the excess flow of the vertex u and the capacity of the edge
                    flow = min(self.vertices[u].e, self.edges[i].capacity - self.edges[i].flow)
                    #The excess flow of the vertex u is decreased by the flow
                    self.vertices[u].e -= flow
                    #The excess flow of the vertex v is increased by the flow
                    self.vertices[self.edges[i].v].e += flow
                    #The flow of the edge is increased by the flow
                    self.edges[i].flow += flow
                    #The reverse flow is updated
                    self.updateReverseFlow(i, flow)

                    return True
        return False
    
    def relabel(self, u):
        """
        Relabels the height of a vertex to the minimum height of its neighbors plus one.
        
        Args:
            u (int): The vertex to be relabeled.
        """
        
        min_height = float('inf')

        for i in range(len(self.edges)):
            #If the edge starts at the vertex u
            if self.edges[i].u == u:
                #If the edge is already at full capacity, continue
                if self.edges[i].flow == self.edges[i].capacity:
                    continue
                #If the height of the vertex v is less than the minimum height, update minimun height
                if self.vertices[self.edges[i].v].h < min_height:
                    min_height = self.vertices[self.edges[i].v].h
        #Relabel vertex u
        self.vertices[u].h = min_height + 1
    
    def push_relabel(self, s, t):   
        """
        Implements the push-relabel algorithm to compute the maximum flow.
        
        Args:
            s (int): The source vertex.
            t (int): The target vertex.
        """

        #Starts the preflow
        self.init_preflow(s)
        #Finds the next vertex with excess flow
        u = self.nextFlowVertex()

        while u != -1:
            #If no push is possible, relabel the vertex
            if not self.push(u):
                self.relabel(u)
            #Finds the next vertex with excess flow
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
        for node1 in self.edges:
            if node1.capacity > 0:
                if node1.flow > 0:
                    s += f"{node1.u} {node1.v} {node1.flow}\n"
        s += f"\nMaximum Flow: {max_flow}\n"
        s += f"Execution Time: {end_time - start_time:.6f} seconds\n"
        return s
    
def read_graph_from_file_push_relabel(input_file):
        """
        Reads graph edges from a .in file for the push relabel algorithm.

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
    push_relable_output = graph_push_relabel.structure_output(0, int(V-1))
    #pr_output = graph_push_relabel.structure_output("0", str(max(int(node) for node in graph_push_relabel.nodes)))
    
    with open(output_file, "w") as f:
        f.write("Algorithm Results:\n\n")
        f.write("Edmonds-Karp Algorithm\n")
        f.write(ek_output)
        f.write("\n")
        f.write("Push-Relabel Algorithm\n")
        f.write(push_relable_output)
    
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python maximum_flow.py <input_file> <output_file>")
        sys.exit(1)
    run_algorithms(sys.argv[1], sys.argv[2])