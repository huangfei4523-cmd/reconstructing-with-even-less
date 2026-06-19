"""测试 EdgePredictionGNN 的前向传播和边预测。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import torch
from gnn_model import EdgePredictionGNN


class TestEdgePrediction(unittest.TestCase):

    def setUp(self):
        self.model = EdgePredictionGNN(feature_dim=16, hidden_dim=64, emb_dim=32)
        self.N = 10

    def test_forward_output_shape(self):
        """验证 forward 的 node_emb 和 edge_logits shape"""
        node_feat = torch.randn(self.N, 16)
        edges = []
        for i in range(self.N):
            for j in range(i + 1, min(i + 4, self.N)):
                edges.append([i, j])
                edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).T

        node_emb, edge_logits = self.model(node_feat, edge_index, None)
        self.assertEqual(node_emb.shape, (self.N, 32))
        self.assertEqual(edge_logits.shape, (edge_index.shape[1],))

    def test_forward_with_edge_features(self):
        """验证带边特征的 forward"""
        node_feat = torch.randn(self.N, 16)
        E = 15
        edge_index = torch.randint(0, self.N, (2, E), dtype=torch.long)
        edge_feat = torch.randn(E, 4)

        node_emb, edge_logits = self.model(node_feat, edge_index, edge_feat)
        self.assertEqual(edge_logits.shape, (E,))

    def test_predict_all_pairs_symmetry(self):
        """验证 predict_all_pairs 输出的对称性"""
        node_emb = torch.randn(self.N, 32)
        probs = self.model.predict_all_pairs(node_emb, batch_size=5)
        self.assertEqual(probs.shape, (self.N, self.N))
        # 对称性
        self.assertTrue(torch.allclose(probs, probs.T, atol=1e-4))

    def test_predict_all_pairs_range(self):
        """验证概率值在 [0, 1]"""
        node_emb = torch.randn(self.N, 32)
        probs = self.model.predict_all_pairs(node_emb, batch_size=5)
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())


if __name__ == "__main__":
    unittest.main()
