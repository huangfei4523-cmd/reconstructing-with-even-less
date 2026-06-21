"""测试 Phase 3 力导向布局"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest, numpy as np
from gnn_reconstruction import ForceDirectedLayout, CheckReconstructionFailure


class TestLayout(unittest.TestCase):
    def test_grid_converges(self):
        """连通网格图布局收敛"""
        N = 25
        E_hat = []
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if j < 4:
                    E_hat.append((idx, idx + 1, 0.8))
                if i < 4:
                    E_hat.append((idx, idx + 5, 0.8))
        pos = ForceDirectedLayout(E_hat, N)
        self.assertEqual(pos.shape, (N, 2))
        self.assertGreater(pos.var(), 0)

    def test_failure_detection_few_edges(self):
        """边数不足时检测失败"""
        pos = np.random.rand(10, 2)
        failed, _ = CheckReconstructionFailure(pos, [(0, 1, 0.5)])
        self.assertTrue(failed)


if __name__ == "__main__":
    unittest.main()
