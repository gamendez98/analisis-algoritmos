'''
The following python program finds if a graph is acyclic or not using DFS algorithm
If it is not, then it returns all the cycles present in the graph
In this particular case study, each vertex in the graph represents a bank and each edge represents a loan.
The weight of said edge represents the amount of the loan.
Hence, each cycle represents a self-loan.
'''

import sys

class Graph:
    """
    A class to represent a graph using edges and vertices.

    Attributes:
        edges (list[tuple[str, str, int]]): A list of edges in the format (vertex1, vertex2, weight).
        vertices (set[str]): A set of all unique vertices in the graph.
    """
    def __init__(self):
        """
        Initializes the Graph instance the necessary data structures.
        """
        self.vertex_index = {} #vertex -> index
        self.inversed_vertex_index = {} #index -> vertex
        self.adj_matrix = [] #adjacency matrix
    def add_vertex(self, vertex):
        """
        Adds a vertex to the graph if it does not already exist.
        It also expands the adjacency matrix to accomodate the new vertex.
        
        Args:
            vertex (str): The vertex to add.
        """
        if vertex not in self.vertex_index:
            index = len(self.vertex_index)
            self.vertex_index[vertex] = index
            self.inversed_vertex_index[index] = vertex
            for row in self.adj_matrix:
                row.append(0)
            self.adj_matrix.append([0]*len(self.vertex_index))
    def add_edge(self, vertex1, vertex2, weight):
        """
        Adds an edge to the graph.

        Args:
            vertex1 (str): The first vertex of the edge.
            vertex2 (str): The second vertex of the edge.
            weight (int): The weight of the edge.
        """
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        index1 = self.vertex_index[vertex1]
        index2 = self.vertex_index[vertex2]
        self.adj_matrix[index1][index2] = weight
        

def dfs(graph):
    """
    Performs a Depth-First Search (DFS) on the graph to find cycles.
    
    Args:
        graph (Graph): The graph for which to find the MST.

    Returns:
        list[list[str]]: List of cycles if the graph is not acyclic. If it is, it returns an empty list.
    """
    n = len(graph.vertex_index)
    visited = [False]*n
    recursive_stack = [False]*n
    cycles = []
    def dfs_visit(u, path):
        """
        Missing Doc
        """
        visited[u] = True
        recursive_stack[u] = True
        path.append(u)

        for v in range(n):
            if graph.adj_matrix[u][v] != 0:
                if not visited[v]:
                    dfs_visit(v, path)
                elif recursive_stack[v]: #We found a cycle
                    cycle_start_index = path.index(v)
                    cycle = path[cycle_start_index:]
                    cycle_lables = [graph.inversed_vertex_index[i] for i in cycle]
                    cycles.append(cycle_lables)
        path.pop()
        recursive_stack[u] = False
    for i in range(n):
        if not visited[i]:
            dfs_visit(i, [])
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
    graph = Graph()
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:  # Validate line format
                raise ValueError(f"Invalid line format: {line.strip()}")
            vertex1, vertex2, weight = parts
            weight = int(weight)  # Convert weight to integer
            if weight < 0:  # Check for negative weights
                raise ValueError(f"Negative weight detected: {line.strip()}")
            graph.add_edge(vertex1, vertex2, weight)  # Add the edge to graph
    return graph

def write_cycles_to_file(output_file, cycles):
    """
    Writes the MST edges to a .out file.

    Args:
        output_file (str): The path to the output file.
        mst (list[tuple[str, str, int]]): The MST edges to write.
    """
    with open(output_file, 'w') as f:
        if len(cycles) == 0:
            f.write("No self-loans were detected\n")
        else:
            for cycle in cycles:
                f.write(" -> ".join(cycle) + "\n") # Write each cycle in the format "node1 -> node2 -> ... -> node1\n"



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
        cycles = dfs(graph)  # Compute the MST using Kruskal's algorithm
        write_cycles_to_file(output_file, cycles)  # Write the MST to the output file
    except ValueError as e:
        print(f"Error: {e}")  # Print error message if an exception occurs
        sys.exit(1)

if __name__ == '__main__':
    main()


#Si está en la fila es gris
