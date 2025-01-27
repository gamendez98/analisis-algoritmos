import unittest
from dfs_v2 import Graph, dfs, read_graph_from_file

class TestDFSAlgorithm(unittest.TestCase):

    def test_simple_graph_no_cycles(self):
        edges = [
            ('A', 'B', 1),
            ('B', 'C', 2),
            ('C', 'D', 3)
        ]
        graph = Graph(edges)
        cycles = dfs(graph)
        self.assertEqual(len(cycles), 0)

    def test_simple_graph_with_cycle(self):
        edges = [
            ('A', 'B', 1),
            ('B', 'C', 2),
            ('C', 'A', 3)
        ]
        graph = Graph(edges)
        cycles = dfs(graph)
        expected_cycles = [
            ['A', 'B', 'C']
        ]
        self.assertEqual(len(cycles), 1)
        self.assertTrue(sorted(cycles[0]) == sorted(expected_cycles[0]))

    def test_multiple_cycles(self):
        edges = [
            ('A', 'B', 1),
            ('B', 'C', 2),
            ('C', 'A', 3),
            ('C', 'D', 4),
            ('D', 'E', 5),
            ('E', 'C', 6)
        ]
        graph = Graph(edges)
        cycles = dfs(graph)
        expected_cycles = [
            ['A', 'B', 'C'],
            ['C', 'D', 'E']
        ]
        self.assertEqual(len(cycles), len(expected_cycles))
        for cycle in expected_cycles:
            self.assertTrue(any(sorted(cycle) == sorted(found_cycle) for found_cycle in cycles))

    def test_disconnected_graph(self):
        edges = [
            ('A', 'B', 1),
            ('C', 'D', 2)
        ]
        graph = Graph(edges)
        cycles = dfs(graph)
        self.assertEqual(len(cycles), 0)

    def test_large_graph_with_cycle(self):
        edges = [
            ('A', 'B', 10),
            ('B', 'C', 15),
            ('C', 'D', 20),
            ('D', 'E', 25),
            ('E', 'A', 30),
            ('F', 'G', 35),
            ('G', 'H', 40),
            ('H', 'F', 45)
        ]
        graph = Graph(edges)
        cycles = dfs(graph)
        expected_cycles = [
            ['A', 'B', 'C', 'D', 'E'],
            ['F', 'G', 'H']
        ]
        self.assertEqual(len(cycles), len(expected_cycles))
        for cycle in expected_cycles:
            self.assertTrue(any(sorted(cycle) == sorted(found_cycle) for found_cycle in cycles))

    def test_negative_weight_error(self):
        with self.assertRaises(ValueError):
            read_graph_from_file("test_negative_weight.in")

if __name__ == "__main__":
    unittest.main()
