import math
import random
from functools import lru_cache


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
        self.nodes = list(G.nodes())
        self.out_w = {
            u: sum(d.get('weight', 1) for _, _, d in G.out_edges(u, data=True))
            for u in self.nodes
        }
        self.succ = {
            u: [(v, d.get('weight', 1)) for _, v, d in G.out_edges(u, data=True)]
            for u in self.nodes
        }
        self.modules = {}
        self.cached_get_module_entropy_info = {}

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
                out_edges = self.succ[node]
                if not out_edges:
                    # Dangling node: no outlinks. Distribute its (follow) probability uniformly.
                    contribution = (1 - self.teleportation) * p[node] / N
                    for dest in self.G.nodes():
                        new_p[dest] += contribution
                else:
                    # Total weight of node's out-going links.
                    total_w = sum(weight for neighbor, weight in out_edges)
                    for neighbor, weight in out_edges:
                        new_p[neighbor] += (1 - self.teleportation) * p[node] * (weight / total_w)

            # Check convergence (L1 norm of difference)
            err = sum(abs(new_p[n] - p[n]) for n in self.G.nodes())
            p = new_p
            if err < self.tol:
                break

        sum_p = sum(p.values())
        assert abs(sum_p - 1.0) < 1e-6, f"Stationary dist not normalized: sum={sum_p}"
        return p

    def update_module(self, partition, node, module_id):
        old_module = partition[node]
        partition[node] = module_id
        self.invalidate_cached_get_module_entropy_info(old_module)
        self.invalidate_cached_get_module_entropy_info(module_id)
        self.modules[module_id].add(node)
        self.modules.setdefault(old_module, set()).discard(node)
        return partition

    def invalidate_cached_get_module_entropy_info(self, module_id):
        self.cached_get_module_entropy_info.pop(module_id, None)

    def get_module_entropy_info(self, module_id, nodes, partition, p):
        cached_info = self.cached_get_module_entropy_info.get(module_id, None)
        if cached_info is not None:
            return cached_info
        info = self.compute_module_entropy_info(module_id, nodes, partition, p)
        self.cached_get_module_entropy_info[module_id] = info
        return info

    def compute_module_entropy_info(self, module_id, nodes, partition, p):
        pm = sum(p[n] for n in nodes)
        ex = 0.0
        for n in nodes:
            w_out = self.out_w[n]
            if w_out == 0:
                continue
            for nbr, w in self.succ[n]:
                if partition[nbr] != module_id:
                    ex += p[n] * (1 - self.teleportation) * (w / w_out)
        return pm, ex

    def set_modules(self, partition):
        self.modules = {}
        for n in self.nodes:  # no .items() overhead
            self.modules.setdefault(partition[n], set()).add(n)

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
        log2 = math.log2

        # 1) group nodes
        modules = {}
        for n in self.nodes:  # no .items() overhead
            modules.setdefault(partition[n], []).append(n)

        # 2) aggregate per‑module stats
        p_module = {}
        exit_prob = {}
        for m, nodes in modules.items():
            pm, ex = self.get_module_entropy_info(m, nodes, partition, p)
            p_module[m] = pm
            exit_prob[m] = ex

        q_exit = sum(exit_prob.values())

        # 3) inter‑module entropy
        H_exit = 0.0
        if q_exit:
            inv_q = 1.0 / q_exit
            for ex in exit_prob.values():
                if ex:  # skip zeros
                    prob = ex * inv_q
                    H_exit -= prob * log2(prob)

        # 4) within‑module contribution
        L_within = 0.0
        for m, nodes in modules.items():
            pm, ex = p_module[m], exit_prob[m]
            tot = pm + ex
            if tot == 0:
                continue
            Hm = 0.0
            if ex:
                pe = ex / tot
                Hm -= pe * log2(pe)
            inv_tot = 1.0 / tot
            for n in nodes:
                pn = p[n] * inv_tot
                Hm -= pn * log2(pn)
            L_within += tot * Hm

        return q_exit * H_exit + L_within

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
        self.set_modules(partition)
        # Compute the initial description length.
        current_L = self.compute_map_equation(partition, p)

        improvement = True
        it = 0
        while improvement and it < self.max_iter:
            improvement = False
            it += 1
            nodes = list(self.G.nodes())
            random.shuffle(nodes)  # Process nodes in random order to reduce bias.
            for node in nodes:
                current_module = partition[node]
                candidates = {partition[v] for v, _ in self.succ[node]} | {partition[u] for u, _ in
                                                                           self.G.in_edges(node)}
                candidates.add(current_module)

                best_L = current_L
                best_module = current_module
                for module in candidates:
                    if module == current_module:
                        continue
                    old_module = partition[node]
                    partition = self.update_module(partition, node, module)
                    new_L = self.compute_map_equation(partition, p)
                    if new_L < best_L:
                        best_L = new_L
                        best_module = module
                    partition = self.update_module(partition, node, old_module)
                if best_module != current_module:
                    partition = self.update_module(partition, node, best_module)
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
                for j in range(i + 1, len(modules)):
                    m1, m2 = modules[i], modules[j]
                    # Temporarily merge m2 into m1
                    backup = partition.copy()
                    for n, m in backup.items():
                        if m == m2:
                            partition = self.update_module(partition, n, m1)
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
                        partition = self.update_module(partition, n, m1)
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
