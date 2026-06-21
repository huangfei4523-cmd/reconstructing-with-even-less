"""测试 Phase 1 训练数据生成"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest, numpy as np
from gnn_model import generate_training_data_v2


class TestPhase1Data(unittest.TestCase):
    def test_small_sample_generation(self):
        """3种 N × 2种形状 × 2种 p 均能成功生成"""
        for n in [20, 40, 200]:
            for p in [0.01, 0.50]:
                configs = [
                    {"type": "grid_2d", "ratio": 1.0, "grid": (min(n, 20), min(n, 20)),
                     "density": 1, "ratio_resp": p},
                    {"type": "random_2d", "ratio": 1.0, "N_points": n,
                     "range": (30, 30), "ratio_resp": p},
                ]
                feat, adj, resp = generate_training_data_v2(num_samples=4, configs=configs)
                self.assertGreaterEqual(len(feat), 2)
                for f, a in zip(feat, adj):
                    self.assertEqual(f.shape[1], 16)
                    self.assertEqual(a.shape, (f.shape[0], f.shape[0]))

    def test_all_configs_generate(self):
        """默认配置能生成完整样本"""
        feat, adj, resp = generate_training_data_v2(num_samples=10, configs=None)
        self.assertEqual(len(feat), 10)
        for f, a in zip(feat, adj):
            self.assertGreaterEqual(f.shape[0], 4)
            self.assertTrue(a.sum() > 0, "至少有一条正样本边")


if __name__ == "__main__":
    unittest.main()
