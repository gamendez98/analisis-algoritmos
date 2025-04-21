import math
import random

class Infomap:
    """
    Infomap algorithm for community detection on weighted directed graphs.
    Encapsulates the steps:
      1) Compute stationary distribution
      2) Greedy node moves
      3) Module merging
      4) Final map equation computation (description length L)

    Complexity:
    Let N=|V|, E=|E|, modules=number of modules, max_iter=max iterations.
      - compute_stationary_distribution: O(max_iter * (N + E))
      - compute_map_equation: O(N + E)
      - infomap_greedy: O(max_iter * N * (N + E))
      - merge_modules: O(modules^2 * (N + E)) per full merge phase
      - run (full pipeline): dominated by greedy & merge steps: O(max_iter * N * (N + E) + modules^2 * (N + E))
    """
    def __init__(self, G, teleportation=0.15, max_iter=100, tol=1e-8):
        """
        Initializes the Infomap instance.

        Parameters:
          G: A NetworkX DiGraph with optional 'weight' on edges.
          teleportation: The teleportation probability (typical 0.15).
          max_iter: Maximum number of iterations for greedy moves and power method.
          tol: Convergence tolerance for the stationary distribution.
        """
        self.G = G
        self.teleportation = teleportation
        self.max_iter = max_iter
        self.tol = tol

    def compute_stationary_distribution(self):
        """
        Compute stationary distribution via power method with teleportation.

        Returns:
          dict mapping node -> probability.
        Complexity: O(max_iter * (N + E))
        """
        assert 0 < self.teleportation < 1, "Teleportation must be in (0, 1)"
        N = self.G.number_of_nodes()

        # Initialize with a uniform distribution
        p = {node: 1.0 / N for node in self.G.nodes()}

        for _ in range(self.max_iter):
            # Start with the teleportation contribution for every node.
            new_p = {node: self.teleportation / N for node in self.G.nodes()}
            # For each node, distribute its probability mass along its outgoing edges.
            for node in self.G.nodes():
                # Get all outgoing edges from node along with their weight.
                out_edges = list(self.G.out_edges(node, data=True))
                if not out_edges:
                    # Dangling node: no outlinks. Distribute its (follow) probability uniformly.
                    contribution = (1 - self.teleportation) * p[node] / N
                    for dest in self.G.nodes():
                        new_p[dest] += contribution
                else:
                    # Total weight of node's out-going links.
                    total_w = sum(data.get('weight', 1) for (_, _, data) in out_edges)
                    for (_, neighbor, data) in out_edges:
                        weight = data.get('weight', 1)
                        new_p[neighbor] += (1 - self.teleportation) * p[node] * (weight / total_w)

            # Check convergence (L1 norm of difference)
            err = sum(abs(new_p[n] - p[n]) for n in self.G.nodes())
            p = new_p
            if err < self.tol:
                break

        sum_p = sum(p.values())
        assert abs(sum_p - 1.0) < 1e-6, f"Stationary dist not normalized: sum={sum_p}"
        return p

    def compute_map_equation(self, partition, p):
        """
        Compute the map equation L for a given partition and stationary dist p.

        Parameters:
          partition: dict mapping node -> module ID
          p: stationary distribution dict
        Returns:
          L_total: description length (bits)
        Complexity: O(N + E)
        """
        # Group nodes by module
        modules = {}
        for node, module in partition.items():
            modules.setdefault(module, []).append(node)

        # Precompute out-weight
        out_weight = {}
        for node in self.G.nodes():
            edges = list(self.G.out_edges(node, data=True))
            out_weight[node] = sum(data.get('weight', 1) for (_, _, data) in edges) if edges else 0

        # For each module, compute the total node probability and the exit probability.
        module_info = {}
        for module, nodes in modules.items():
            p_module = sum(p[n] for n in nodes)
            exit_module = 0.0
            # For each node in the module, add the probability that the random walker leaves the module.
            for node in nodes:
                if out_weight[node] > 0:
                    for _, neighbor, data in self.G.out_edges(node, data=True):
                        if partition[neighbor] != module:
                            weight = data.get('weight', 1)
                            exit_module += p[node] * (1 - self.teleportation) * (weight / out_weight[node])
            module_info[module] = {'p_module': p_module, 'exit': exit_module, 'nodes': nodes}

        # Total exit probability (the inter-module flow) summed over all modules.
        q_exit = sum(info['exit'] for info in module_info.values())

        # Inter-module entropy
        H_exit = 0.0
        if q_exit > 0:
            for info in module_info.values():
                e = info['exit']
                if e > 0:
                    prob = e / q_exit
                    H_exit -= prob * math.log(prob, 2)

        # Within-module contribution
        L_within = 0.0
        for info in module_info.values():
            total_flow = info['p_module'] + info['exit']
            H_module = 0.0
            if info['exit'] > 0:
                prob_exit = info['exit'] / total_flow
                H_module -= prob_exit * math.log(prob_exit, 2)
            for node in info['nodes']:
                if p[node] > 0:
                    prob_node = p[node] / total_flow
                    H_module -= prob_node * math.log(prob_node, 2)
            L_within += total_flow * H_module

        assert abs(sum(info['p_module'] for info in module_info.values()) - 1.0) < 1e-6, \
               "Conservation violated: sum of p_module != 1"
        L_total = q_exit * H_exit + L_within
        return L_total

    def infomap_greedy(self, p):
        """
        Perform greedy node moves to optimize L.

        Parameters:
          p: stationary distribution dict
        Returns:
          partition: dict mapping node -> module
          current_L: final description length
        Complexity: O(max_iter * N * (N + E))
        """
        # Initialize partition: each node in its own module.
        partition = {n: n for n in self.G.nodes()}
        # Compute the initial description length.
        current_L = self.compute_map_equation(partition, p)

        improvement = True
        it = 0
        while improvement and it < self.max_iter:
            improvement = False
            it += 1
            nodes = list(self.G.nodes())
            random.shuffle(nodes) # Process nodes in random order to reduce bias.
            for node in nodes:
                current_module = partition[node]
                candidates = {partition[v] for _, v in self.G.out_edges(node)} | {partition[u] for u, _ in self.G.in_edges(node)}
                candidates.add(current_module)

                best_L = current_L
                best_module = current_module
                for module in candidates:
                    if module == current_module:
                        continue
                    old_module = partition[node]
                    partition[node] = module
                    new_L = self.compute_map_equation(partition, p)
                    if new_L < best_L:
                        best_L = new_L
                        best_module = module
                    partition[node] = old_module
                if best_module != current_module:
                    partition[node] = best_module
                    current_L = best_L
                    improvement = True

        return partition, current_L

    def merge_modules(self, partition, p):
        """
        Greedily merge entire modules if L decreases.

        Parameters:
          partition: dict mapping node -> module
          p: stationary distribution dict
        Returns:
          partition: updated module mapping
        Complexity: O(modules^2 * (N + E))
        """
        improved = True
        while improved:
            improved = False
            modules = list(set(partition.values()))
            curr_L = self.compute_map_equation(partition, p)
            best_delta = 0
            best_pair = None
            for i in range(len(modules)):
                for j in range(i+1, len(modules)):
                    m1, m2 = modules[i], modules[j]
                    # Temporarily merge m2 into m1
                    backup = partition.copy()
                    for n, m in backup.items():
                        if m == m2:
                            partition[n] = m1
                    new_L = self.compute_map_equation(partition, p)
                    delta = curr_L - new_L
                    # Revert the temporary merge
                    partition = backup
                    if delta > best_delta:
                        best_delta = delta
                        best_pair = (m1, m2)
            if best_pair:
                m1, m2 = best_pair
                for n, m in partition.items():
                    if m == m2:
                        partition[n] = m1
                improved = True
        return partition

    def run(self):
        """
        Execute full Infomap pipeline.

        Returns:
          partition: final node → module mapping
          L_final: final description length
        Complexity: O(max_iter * N * (N + E) + modules^2 * (N + E))
        """
        p = self.compute_stationary_distribution()
        partition, _ = self.infomap_greedy(p)
        partition = self.merge_modules(partition, p)
        L_final = self.compute_map_equation(partition, p)
        return partition, L_final