"""测试消息传递图构建。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from gnn_model import build_message_passing_graph, build_message_passing_graph_from_features


class TestBuildMessagePassingGraph(unittest.TestCase):

    def test_sparse_format(self):
        """验证输出为 [2, E] 稀疏格式"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(cooc, 0)
        edge_index = build_message_passing_graph(cooc, k=5)
        self.assertEqual(len(edge_index.shape), 2)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertGreater(edge_index.shape[1], 0)

    def test_no_self_loops(self):
        """验证图中无自环"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(cooc, 0)
        edge_index = build_message_passing_graph(cooc, k=3)
        for e in range(edge_index.shape[1]):
            self.assertNotEqual(edge_index[0, e], edge_index[1, e],
                                f"Self-loop at edge {e}")

    def test_edge_count(self):
        """验证 k-NN 边数至少 k*N（对称化前），对称化后更多"""
        N = 20
        cooc = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(cooc, 0)
        edge_index = build_message_passing_graph(cooc, k=3)
        self.assertGreaterEqual(edge_index.shape[1], 3 * N)

    def test_from_features(self):
        """验证 build_message_passing_graph_from_features 也输出正确格式"""
        N = 10
        node_feat = np.random.rand(N, 16).astype(np.float32)
        edge_index = build_message_passing_graph_from_features(node_feat, k=5)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertGreater(edge_index.shape[1], 0)
        for e in range(edge_index.shape[1]):
            self.assertNotEqual(edge_index[0, e], edge_index[1, e])

    def test_small_n(self):
        """N 小时不报错"""
        for N in [1, 2, 3]:
            cooc = np.random.rand(N, N).astype(np.float32)
            edge_index = build_message_passing_graph(cooc, k=3)
            self.assertEqual(edge_index.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
