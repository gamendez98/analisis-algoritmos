import networkx as nx
import random

# Optional: seed for reproducibility
# random.seed(42)

class GeneticAlgorithmCommunityDetection:
    """
    Genetic algorithm for community detection via modularity optimization.

    Overall Complexity:
      O(G * P * (N + E)), where
        G = generations,
        P = population_size,
        N = number of nodes,
        E = number of edges.
    """
    def __init__(self,
                 G,
                 population_size=50,
                 generations=100,
                 mutation_rate=0.1,
                 tournament_size=3,
                 elitism=True,
                 early_stop=True,
                 no_improve_gen=30):
        """
        Initialize GA parameters.

        Parameters:
          G: NetworkX Graph (undirected, weighted).
          population_size: number of candidate partitions per generation.
          generations: number of generations to evolve.
          mutation_rate: probability of mutating each label.
          tournament_size: competitors in selection.
          elitism: carry best individual to next generation.
          early_stop: halt if no improvement.
          no_improve_gen: gens without improvement to stop.
        """
        self.G = G
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.early_stop = early_stop
        self.no_improve_gen = no_improve_gen

    @staticmethod
    def partition_to_communities(partition):
        """
        Group nodes by community label.

        Complexity: O(N), where N = number of nodes.
        """
        communities = {}
        for node, label in partition.items():  # O(N)
            communities.setdefault(label, set()).add(node)
        return list(communities.values())

    def compute_modularity(self, partition):
        """
        Compute weighted modularity of a partition.

        Uses NetworkX modularity.
        Complexity: O(N + E).
        """
        communities = self.partition_to_communities(partition)  # O(N)
        return nx.algorithms.community.quality.modularity(
            self.G, communities, weight='weight'
        )  # O(E + N)

    def initialize_population(self):
        """
        Create initial random population of partitions.

        Complexity: O(P * N), P = population_size.
        """
        nodes = list(self.G.nodes())  # O(N)
        n = len(nodes)
        population = []
        for _ in range(self.population_size):  # P iterations
            individual = {node: random.randint(0, n-1) for node in nodes}  # O(N)
            population.append(individual)
        return population

    def tournament_selection(self, population, fitnesses):
        """
        Select one parent via tournament.

        Complexity: O(tournament_size).
        """
        selected = random.sample(range(len(population)), self.tournament_size)
        best = max(selected, key=lambda i: fitnesses[i])
        return population[best]

    def uniform_crossover(self, parent1, parent2):
        """
        Uniform crossover between two parents.

        Complexity: O(N).
        """
        child = {}
        for node in parent1:  # O(N)
            child[node] = parent1[node] if random.random() < 0.5 else parent2[node]
        return child

    def mutate(self, individual):
        """
        Mutate an individual’s labels.

        Complexity: O(N).
        """
        mutated = individual.copy()  # O(N)
        label_range = len(self.G.nodes())
        for node in mutated:  # O(N)
            if random.random() < self.mutation_rate:
                mutated[node] = random.randint(0, label_range-1)
        return mutated

    def run(self):
        """
        Execute the GA evolution.

        Returns:
          best_partition: dict mapping node -> label
          best_modularity: float

        Per-generation Complexity:
          Selection:      O(P * T)
          Crossover:      O(P * N)
          Mutation:       O(P * N)
          Fitness eval:   O(P * (N + E))
        Overall: O(G * P * (N + E)).
        """
        population = self.initialize_population()  # O(P*N)
        fitnesses = [self.compute_modularity(ind) for ind in population]  # O(P*(N+E))
        best_index = max(range(self.population_size), key=lambda i: fitnesses[i])
        best_ind = population[best_index].copy()
        best_fit = fitnesses[best_index]
        no_improve = 0

        for gen in range(self.generations):
            new_pop = []
            if self.elitism:
                new_pop.append(best_ind.copy())

            while len(new_pop) < self.population_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                child = self.uniform_crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)

            population = new_pop
            fitnesses = [self.compute_modularity(ind) for ind in population]  # O(P*(N+E))

            current_best = max(range(self.population_size), key=lambda i: fitnesses[i])
            if fitnesses[current_best] > best_fit:
                best_fit = fitnesses[current_best]
                best_ind = population[current_best].copy()
                no_improve = 0
            else:
                no_improve += 1

            print(f"Generation {gen}: Best modularity = {best_fit:.4f}")

            if self.early_stop and no_improve >= self.no_improve_gen:
                print(f"No improvement for {self.no_improve_gen} generations; stopping early.")
                break

        return best_ind, best_fit