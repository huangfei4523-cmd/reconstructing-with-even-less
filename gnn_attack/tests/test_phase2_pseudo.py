"""测试 Phase 2 伪标签筛选"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest, numpy as np
from gnn_self_training import SelectPseudoLabels


class TestPseudoLabels(unittest.TestCase):
    def test_counts(self):
        """伪标签数量正确"""
        N = 50
        prob = np.random.rand(N, N)
        prob = (prob + prob.T) / 2
        np.fill_diagonal(prob, 0)
        pos, neg = SelectPseudoLabels(prob, N)
        self.assertEqual(len(pos), 2 * N)
        self.assertEqual(len(neg), 10 * N)

    def test_coverage(self):
        """伪正样本覆盖度 ≥ 80%"""
        N = 50
        prob = np.random.rand(N, N)
        prob = (prob + prob.T) / 2
        np.fill_diagonal(prob, 0)
        pos, neg = SelectPseudoLabels(prob, N)
        covered = set()
        for i, j in pos:
            covered.add(i); covered.add(j)
        self.assertGreaterEqual(len(covered), 0.8 * N)


if __name__ == "__main__":
    unittest.main()
