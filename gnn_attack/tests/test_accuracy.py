"""测试 _get_correct_edges_at_scale 精度计算函数。"""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

# gnn_attack 目录与 gnn_attack.py 同名，直接 importlib 加载模块
_ga_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gnn_attack.py")
_spec = importlib.util.spec_from_file_location("gnn_attack_mod", _ga_path)
_ga = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ga)
_get_correct_edges_at_scale = _ga._get_correct_edges_at_scale


class TestCorrectEdges(unittest.TestCase):

    def test_no_self_loops(self):
        """验证返回的边集无自环"""
        # 两个相邻点
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
