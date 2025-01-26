import unittest
from kruskal import Graph, kruskal, read_graph_from_file

class TestKruskalAlgorithm(unittest.TestCase):

    def test_simple_graph(self):
        edges = [
            ('A', 'B', 1),
            ('B', 'C', 2),
            ('A', 'C', 3)
        ]
        graph = Graph(edges)
        mst = kruskal(graph)
        expected_mst = [
            ('A', 'B', 1),
            ('B', 'C', 2)
        ]
        self.assertEqual(sorted(mst), sorted(expected_mst))

    def test_disconnected_graph(self):
        edges = [
            ('A', 'B', 4),
            ('C', 'D', 5)
        ]
        graph = Graph(edges)
        with self.assertRaises(ValueError):
            mst = kruskal(graph)

    def test_large_graph(self):
        edges = [
            ('A', 'B', 10),
            ('B', 'C', 15),
            ('A', 'C', 5),
            ('C', 'D', 10),
            ('D', 'E', 5),
            ('E', 'F', 10),
            ('D', 'F', 15)
        ]
        graph = Graph(edges)
        mst = kruskal(graph)
        expected_mst = [
            ('A', 'C', 5),
            ('D', 'E', 5),
            ('C', 'D', 10),
            ('E', 'F', 10),
            ('A', 'B', 10)
        ]
        self.assertEqual(sorted(mst), sorted(expected_mst))

    def test_negative_weight_error(self):
        with self.assertRaises(ValueError):
            read_graph_from_file("test_negative_weight.in")

if __name__ == "__main__":
    unittest.main()