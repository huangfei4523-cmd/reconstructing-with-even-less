"""
数据集加载工具。
通过 sys.path 导入项目根目录的 dataset.py，避免 exec() 解析源码。
"""

import sys
import os
import pickle

# 将项目根目录加入 sys.path，以便 from dataset import ...
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def _get_dataset_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def load_dataset(name, N0=20, N1=20):
    """
    加载指定数据集。

    Args:
        name: 数据集名称 (cali_50, cali_self, grid, dg, crg, nh, boat)
        N0, N1: grid 数据集的维度

    Returns:
        dict with keys: points, map_to_original, N0, N1, is_3d, N2
    """
    import process_database
    from dataset import cali_all, cali_self, nh

    dataset_dir = _get_dataset_dir()

    if name == "cali_50":
        print(f"  加载 cali_50 数据集: {len(cali_all)} 个点")
        cali_scaled = process_database.scale_points(cali_all, 50, 50)
        points, map_to_original, n0, n1 = process_database.make_database_from_points(cali_scaled)
        return {
            "points": points,
            "map_to_original": map_to_original,
            "N0": n0,
            "N1": n1,
            "is_3d": False,
        }

    elif name == "cali_self":
        print(f"  加载 cali_self 数据集: {len(cali_self)} 个点")
        cali_scaled = process_database.scale_points(cali_self, 50, 50)
        points, map_to_original, n0, n1 = process_database.make_database_from_points(cali_scaled)
        return {
            "points": points,
            "map_to_original": map_to_original,
            "N0": n0,
            "N1": n1,
            "is_3d": False,
        }

    elif name == "grid":
        print(f"  生成 grid 数据集: {N0}x{N1}")
        points, map_to_original = process_database.get_random_database(N0, N1, 1)
        return {
            "points": points,
            "map_to_original": map_to_original,
            "N0": N0,
            "N1": N1,
            "is_3d": False,
        }

    elif name == "dg":
        path = os.path.join(dataset_dir, "dg")
        print(f"  加载 dg 数据集: {path}")
        with open(path, "rb") as f:
            schools_small = pickle.load(f)
        points, map_to_original, n0, n1 = process_database.make_database_from_points(schools_small)
        return {
            "points": points,
            "map_to_original": map_to_original,
            "N0": n0,
            "N1": n1,
            "is_3d": False,
        }

    elif name == "crg":
        path = os.path.join(dataset_dir, "crg")
        print(f"  加载 crg 数据集: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        points, map_to_original, n0, n1, n2 = process_database.make_database_from_points_3D(data)
        return {
            "points": points,
            "map_to_original": map_to_original,
            "N0": n0,
            "N1": n1,
            "N2": n2,
            "is_3d": True,
        }

    elif name == "nh":
        print(f"  加载 nh 数据集: {len(nh)} 个点")
        points, map_to_original, n0, n1, n2 = process_database.make_database_from_points_3D(nh)
        return {
            "points": points,
            "map_to_original": map_to_original,
            "N0": n0,
            "N1": n1,
            "N2": n2,
            "is_3d": True,
        }

    elif name == "boat":
        path = os.path.join(dataset_dir, "boat.pickle")
        print(f"  加载 boat 数据集: {path}")
        with open(path, "rb") as f:
            boat = pickle.load(f)
        points, map_to_original, n0, n1 = process_database.make_database_from_points(boat)
        return {
            "points": points,
            "map_to_original": map_to_original,
            "N0": n0,
            "N1": n1,
            "is_3d": False,
        }

    else:
        print(f"未知数据集: {name}")
        return {"points": None, "map_to_original": None, "N0": 0, "N1": 0, "is_3d": False}
