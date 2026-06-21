"""测试 _get_correct_edges_at_scale 精度计算函数。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from gnn_model import _get_correct_edges_at_scale


class TestCorrectEdges(unittest.TestCase):

    def test_no_self_loops(self):
        """验证返回的边集无自环"""
        map_to_original = {101: (0, 0), 102: (0, 1)}
        points = [101, 102]
        edges = _get_correct_edges_at_scale(points, map_to_original)
        for e in edges:
            self.assertNotEqual(e[0], e[1], f"Self-loop found: {e}")

    def test_adjacent_points(self):
        """验证正确识别两个相邻点"""
        map_to_original = {101: (0, 0), 102: (0, 1)}
        points = [101, 102]
        edges = _get_correct_edges_at_scale(points, map_to_original)
        self.assertIn(((0, 0), (0, 1)), edges)
        self.assertIn(((0, 1), (0, 0)), edges)

    def test_non_adjacent_points(self):
        """验证不相邻的点对不出现在边集中"""
        map_to_original = {101: (0, 0), 102: (5, 5)}
        points = [101, 102]
        edges = _get_correct_edges_at_scale(points, map_to_original)
        self.assertNotIn(((0, 0), (5, 5)), edges)

    def test_3d_adjacent(self):
        """验证 3D 空间中的邻接点"""
        map_to_original = {101: (0, 0, 0), 102: (0, 0, 1)}
        points = [101, 102]
        edges = _get_correct_edges_at_scale(points, map_to_original)
        self.assertIn(((0, 0, 0), (0, 0, 1)), edges)

    def test_3d_non_adjacent(self):
        """验证 3D 空间中不相邻的点对"""
        map_to_original = {101: (0, 0, 0), 102: (2, 2, 2)}
        points = [101, 102]
        edges = _get_correct_edges_at_scale(points, map_to_original)
        self.assertNotIn(((0, 0, 0), (2, 2, 2)), edges)

    def test_diagonal_not_adjacent(self):
        """验证对角点不算相邻（曼哈顿距离=2）"""
        map_to_original = {101: (0, 0), 102: (1, 1)}
        points = [101, 102]
        edges = _get_correct_edges_at_scale(points, map_to_original)
        self.assertNotIn(((0, 0), (1, 1)), edges)


if __name__ == "__main__":
    unittest.main()
