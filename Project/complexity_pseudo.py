

def run(): # O(max(edges, n) * n * max_iterations)
    initialize_partition() # O(n)
    for iteration in self.max_iterations: # O(max(edges, n) * n * max_iterations)
        for node in nodes: # O(max(edges*n, n**2))
            find_best_comunity(n) # O(max(edges(node)*n, n**2))
        force_split() # O(edges)
        merge_small_comunities() # O(n)
    renumber_communities() # O(n)

def initialize_partition(): # O(n)
    for n in nodes: # O(n)
        pass # O(1)

def find_best_comunity(node): # O(max(edges(node)*n, n**2))
    calculate_node_utility(node) # Overshadowed
    candidate_comunities = set() # O(1)
    for neighbor in self.G.neighbors(node): # O(edges(node))
        candidate_comunities.add(...) # O(1)
    for comunity in candidate_comunities: # O(max(edges(node)*n, n**2)) -> the length of all the comunities ads up to n
        calculate_node_utility(node) # O(max(edges(node), n))
    singleton_utility = sum(... for neighbour in self.G.neighbors(node)) # O(edges(node))


def calculate_node_utility(node): # O(max(edges(node), n))
    for neighbor in self.G.neighbors(node): # O(edges(node))
        pass # O(1)
    comm_size = sum(... for n,c in partition.items()) # O(n)

def force_split(): # O(edges)
    for neighbor in self.G.neighbors(): # O(edges)
        pass # O(1)

def merge_small_comunities(): # O(n)
    communities = defaultdict(list) # O(1)
    for node, comm_id in partition.items(): # O(n)
        communities[comm_id].append(node) # O(1)
    new_partition = partition.copy() # O(n)
    for comm_id, nodes in communities.items(): # O(n)
        if len(nodes) < min_size:
            for node in nodes: # ...
                for neighbor in self.G.neighbors(node): # ...
                    pass # O(1)
            for target_comm in neighbor_comms:
                total_utility = 0
                for node in nodes:
                    total_utility += self.calculate_node_utility(node, target_comm, partition) # O(max(edges(node), n))

            for node in nodes: # O(n)
                new_partition[node] = best_comm # O(1)
    

