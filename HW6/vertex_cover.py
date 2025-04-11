import random
import sys
import time
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import pandas as pd

# For doing insertion and deletion in O(1)
class RandomizedSet:
    def __init__(self):
        self.lst = []
        self.idx_map = {}

    def add(self, val):
        if val in self:
            return False
        self.lst.append(val)
        self.idx_map[val] = len(self.lst) - 1
        return True

    def remove(self, val):
        if val not in self:
            return False
        idx = self.idx_map[val]
        self.lst[idx] = self.lst[-1]
        self.idx_map[self.lst[-1]] = idx
        self.lst.pop()
        del self.idx_map[val]
        return True

    def get_random(self):
        return random.choice(self.lst)

    def __bool__(self):
        return bool(self.lst)

    def __contains__(self, val):
        return val in self.idx_map

class Graph:
    """Simple graph."""
    def __init__(self):
        self.adjacencies = defaultdict(set)
        self.vertices = set()
        self.edges = RandomizedSet()

    def add_edge(self, u, v):
        """Adds an edge between vertices u and v, avoiding self-loops and duplicates."""
        if u == v:
            return
        edge = tuple(sorted((u, v))) 
        if edge not in self.edges:
            self.edges.add(edge)
            self.adjacencies[u].add(v)
            self.adjacencies[v].add(u)
            self.vertices.update({u, v})

    def remove_edges_with_vertex(self, vertex):
        """Removes all edges connected to the given vertex."""
        for neighbor in self.adjacencies.pop(vertex, set()):
            edge = tuple(sorted((vertex, neighbor)))
            self.edges.remove(edge)
            self.adjacencies[neighbor].remove(vertex)

    def get_random_edge(self):
        """Returns a random edge from the graph, or None if empty."""
        return self.edges.get_random() if self.edges.lst else None

    def degree(self, u):
        """Returns the degree of vertex u."""
        return len(self.adjacencies.get(u, []))

    def __repr__(self):
        vertices_str = ', '.join(map(str, sorted(self.vertices)))
        edges_str = '\n'.join([f'{u} - {v}' for u, v in sorted(self.edges.lst)])
        return f'Graph(Vertices: [{vertices_str}], Edges:\n{edges_str})'

def read_graph(filename):
    """Reads a graph from a file and returns a Graph object."""
    graph = Graph()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                u, v = map(int, parts)
                graph.add_edge(u, v)
            except ValueError:
                continue
    return graph

def algorithm_1(graph):
    """Randomly selects an edge, includes both vertices, and removes all connected edges."""
    graph = deepcopy(graph)
    cover = set()
    
    while graph.edges:
        u, v = graph.get_random_edge()
        cover.update({u, v})
        graph.remove_edges_with_vertex(u)
        graph.remove_edges_with_vertex(v)
    
    return cover

def algorithm_2(graph):
    """Selects the vertex with the highest degree, removes its edges, and repeats."""
    graph = deepcopy(graph)
    cover = set()
    
    while graph.edges:
        max_vertex = max(graph.adjacencies, key=graph.degree)
        cover.add(max_vertex)
        graph.remove_edges_with_vertex(max_vertex)
    
    return cover

def algorithm_3(graph):
    """Randomly selects an edge, picks the vertex with the highest degree, and removes its edges."""
    graph = deepcopy(graph)
    cover = set()
    
    while graph.edges:
        u, v = graph.get_random_edge()
        chosen_vertex = u if graph.degree(u) >= graph.degree(v) else v
        cover.add(chosen_vertex)
        graph.remove_edges_with_vertex(chosen_vertex)
    
    return cover

def algorithm_4(graph):
    """Randomly selects an edge, picks a random vertex from it, and removes its edges."""
    graph = deepcopy(graph)
    cover = set()
    
    while graph.edges:
        u, v = graph.get_random_edge()
        chosen_vertex = random.choice([u, v])
        cover.add(chosen_vertex)
        graph.remove_edges_with_vertex(chosen_vertex)
    
    return cover

def experiment(repetitions=5):
    """Runs experiments on graphs of different sizes and densities, averaging over multiple runs."""
    results = []
    sizes = [100, 1000, 10000]
    densities = [0.01, 0.05, 0.1]
    
    for size in sizes:
        print(f"|V| = {size}")
        for density in densities:
            print(f"Density: {density}")
            G = nx.erdos_renyi_graph(size, density)
            graph = Graph()
            for u, v in G.edges():
                graph.add_edge(u, v)
            
            for algo_name, algo in [("Algorithm 1", algorithm_1), ("Algorithm 2", algorithm_2),
                                    ("Algorithm 3", algorithm_3), ("Algorithm 4", algorithm_4)]:
                print(f"Running {algo_name}")
                total_time = 0
                total_cover_size = 0
                for _ in range(repetitions):
                    start_time = time.time()
                    cover = algo(graph)
                    total_time += time.time() - start_time
                    total_cover_size += len(cover)
                
                avg_time = total_time / repetitions
                avg_cover_size = total_cover_size / repetitions
                results.append([size, density, algo_name, avg_cover_size, avg_time])
        print()
    df = pd.DataFrame(results, columns=["Graph Size", "Density", "Algorithm", "Avg Cover Size", "Avg Time (s)"])
    df.to_csv("experiment_results.csv", index=False)
    print(df)

def main():
    """Main function to read input, execute the selected algorithm, and print the result."""
    if len(sys.argv) == 2 and sys.argv[1] == "experiment":
        experiment()
        return
    
    if len(sys.argv) != 3:
        print("Usage: python vertex_cover.py filename.txt algorithm_number (1-4)")
        sys.exit(1)
    
    archivo = sys.argv[1]
    grafo = read_graph(archivo)
    print(grafo)
    
    try:
        algoritmo = int(sys.argv[2])
        if algoritmo not in {1, 2, 3, 4}:
            raise ValueError
    except ValueError:
        print("Invalid algorithm number. Must be 1, 2, 3, or 4.")
        sys.exit(1)
    
    algoritmos = {1: algorithm_1, 2: algorithm_2, 3: algorithm_3, 4: algorithm_4}
    resultado = algoritmos[algoritmo](grafo)
    
    print("Vertex cover:", sorted(resultado))
    print("Size of set:", len(resultado))

if __name__ == "__main__":
    main()
