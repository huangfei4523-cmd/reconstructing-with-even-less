"""测试 extract_node_features 和 extract_edge_features。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from gnn_model import extract_node_features, extract_edge_features


class TestExtractNodeFeatures(unittest.TestCase):

    def test_output_shape(self):
        """验证输出 shape 为 [N, 16]"""
        for N in [5, 10, 50]:
            cooc = np.random.rand(N, N).astype(np.float32)
            np.fill_diagonal(cooc, 0)
            feat = extract_node_features(cooc)
            self.assertEqual(feat.shape, (N, 16))
            self.assertEqual(feat.dtype, np.float32)

    def test_small_n_boundary(self):
        """N < 2 时返回全零特征，N>=2 正常计算，不抛出异常"""
        for N in [1]:
            cooc = np.random.rand(N, N).astype(np.float32)
            feat = extract_node_features(cooc)
            self.assertEqual(feat.shape, (N, 16))
            self.assertTrue(np.allclose(feat, 0))
        for N in [2, 3]:
            cooc = np.random.rand(N, N).astype(np.float32)
            feat = extract_node_features(cooc)
            self.assertEqual(feat.shape, (N, 16))
            self.assertFalse(np.allclose(feat, 0))

    def test_features_not_redundant(self):
        """F3, F5, F7 必须有不同值（当数据有差异时）"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32) * 5
        np.fill_diagonal(cooc, 0)
        cooc[0, :5] += 10  # 让部分点有更强的共现
        feat = extract_node_features(cooc)
        # F3(共现稀疏度), F5(偏度), F7(峰度) 应互不相同
        self.assertFalse(np.allclose(feat[:, 2], feat[:, 4], atol=1e-3),
                         "F3 and F5 should not be identical")
        self.assertFalse(np.allclose(feat[:, 2], feat[:, 6], atol=1e-3),
                         "F3 and F7 should not be identical")


class TestExtractEdgeFeatures(unittest.TestCase):

    def test_output_shape(self):
        """验证边特征 shape 为 [E, 4]"""
        N = 10
        cooc = np.random.rand(N, N).astype(np.float32)
        responses = [{0, 1, 2}, {3, 4, 5}, {1, 3, 6}]
        edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        feat = extract_edge_features(cooc, responses, edge_index)
        self.assertEqual(feat.shape, (edge_index.shape[1], 4))

    def test_edge_feature_range(self):
        """边特征值应在合理范围内"""
        N = 5
        cooc = np.ones((N, N), dtype=np.float32)
        responses = [{0, 1}, {1, 2}]
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
        feat = extract_edge_features(cooc, responses, edge_index)
        # E1 归一化共现应在 [0, 1]
        self.assertTrue(np.all(feat[:, 0] >= 0) and np.all(feat[:, 0] <= 1))
        # E2 Jaccard 应在 [0, 1]
        self.assertTrue(np.all(feat[:, 1] >= 0) and np.all(feat[:, 1] <= 1))
        # E3 余弦相似度应在 [-1, 1]
        self.assertTrue(np.all(feat[:, 2] >= -1) and np.all(feat[:, 2] <= 1))


if __name__ == "__main__":
    unittest.main()
