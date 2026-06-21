"""测试共现图消息传递图构建（设计文档 §1.3）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from gnn_model import build_cooc_message_graph


class TestBuildCoocMessageGraph(unittest.TestCase):

    def test_sparse_format(self):
        """验证输出为 [2, E] 稀疏格式，权重为 [E]"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(cooc, 0)
        edge_index, edge_weights = build_cooc_message_graph(cooc)
        self.assertEqual(len(edge_index.shape), 2)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertGreater(edge_index.shape[1], 0)
        self.assertEqual(len(edge_weights.shape), 1)
        self.assertEqual(edge_weights.shape[0], edge_index.shape[1])

    def test_no_self_loops(self):
        """验证图中无自环"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(cooc, 0)
        edge_index, _ = build_cooc_message_graph(cooc)
        for e in range(edge_index.shape[1]):
            self.assertNotEqual(edge_index[0, e], edge_index[1, e],
                                f"Self-loop at edge {e}")

    def test_weights_softmax_normalized(self):
        """验证每个目标节点的入边权重归一化和为 1"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(cooc, 0)
        _, edge_weights = build_cooc_message_graph(cooc)
        edge_index, _ = build_cooc_message_graph(cooc)
        for i in range(N):
            mask = edge_index[1] == i
            if mask.any():
                self.assertAlmostEqual(edge_weights[mask].sum(), 1.0, delta=0.01)

    def test_symmetric_structure(self):
        """验证图结构对称：每条边 (i,j) 有对应的 (j,i)"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(cooc, 0)
        edge_index, _ = build_cooc_message_graph(cooc)
        # 构建边集合检查对称性
        edges_set = set()
        for e in range(edge_index.shape[1]):
            edges_set.add((edge_index[0, e], edge_index[1, e]))
        for (i, j) in edges_set:
            self.assertIn((j, i), edges_set)

    def test_small_n(self):
        """N 小时不报错"""
        for N in [1, 2, 3]:
            cooc = np.random.rand(N, N).astype(np.float32)
            edge_index, edge_weights = build_cooc_message_graph(cooc)
            self.assertEqual(edge_index.shape[0], 2)
            self.assertEqual(len(edge_weights.shape), 1)

    def test_all_zero_cooc(self):
        """全零共现矩阵返回空图"""
        cooc = np.zeros((5, 5), dtype=np.float32)
        edge_index, edge_weights = build_cooc_message_graph(cooc)
        self.assertEqual(edge_index.shape[1], 0)
        self.assertEqual(len(edge_weights), 0)


if __name__ == "__main__":
    unittest.main()
