"""
数据集加载工具。
从原始 attack.py 中提取硬编码的数据集定义，避免重复。
"""

import sys
import os
import importlib.util
import pickle
import random
import re


def _extract_variable_from_py(filepath, var_name):
    """从 Python 文件中提取指定变量。"""
    with open(filepath, "r") as f:
        content = f.read()

    # 找到变量定义的起始位置
    pattern = rf"{var_name}\s*=\s*\["
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"在 {filepath} 中找不到变量 {var_name}")

    start = match.start()
    # 找到匹配的闭合括号
    depth = 0
    in_bracket = False
    end = start
    for i, ch in enumerate(content[start:], start):
        if ch == "[":
            depth += 1
            in_bracket = True
        elif ch == "]":
            depth -= 1
            if in_bracket and depth == 0:
                end = i + 1
                break

    code = content[start:end]
    namespace = {}
    exec(code, namespace)
    return namespace[var_name]


def _get_dataset_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def load_dataset(name, N0=20, N1=20):
    """
    加载指定数据集。

    Args:
        name: 数据集名称 (cali_50, grid, dg, crg, nh, boat)
        N0, N1: grid 数据集的维度

    Returns:
        dict with keys: points, map_to_original, N0, N1, is_3d, N2
    """
    import process_database

    # 找到原始 attack.py 路径
    original_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "reconstructing-with-even-less",
    )
    original_attack = os.path.join(original_dir, "attack.py")

    dataset_dir = _get_dataset_dir()

    if name == "cali_50":
        # 从原 attack.py 中提取 cali_all
        cali_all = _extract_variable_from_py(original_attack, "cali_all")
        print(f"  加载 cali_50 数据集: {len(cali_all)} 个点")
        cali_all = process_database.scale_points(cali_all, 50, 50)
        points, map_to_original, n0, n1 = process_database.make_database_from_points(cali_all)
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
        nh = _extract_variable_from_py(original_attack, "nh")
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
