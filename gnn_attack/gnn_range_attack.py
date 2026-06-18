"""
GNN 增强的 Range Attack 攻击引擎。

与原始 range_attack.py 的核心区别：
  ┌──────────────────────┬─────────────────────────────┐
  │ 原始方法             │ GNN 增强方法                │
  ├──────────────────────┼─────────────────────────────┤
  │ 集合交集放大信息     │ 共现矩阵 → GNN 直接预测边   │
  │ 找大小为2的素数响应  │ 学习所有响应大小的模式      │
  │ 硬阈值（交集非空）   │ 软概率（学习到的决策边界） │
  │ 确定性算法           │ 数据驱动的学习              │
  └──────────────────────┴─────────────────────────────┘

用法:
    import gnn_range_attack
    G, used = gnn_range_attack.general(responses, model_or_path)
"""

import networkx as nx
import numpy as np
import tqdm
import torch
import os

from gnn_model import EdgePredictionGNN, predict_edges_from_cooccurrence


# ─────────────────────────────────────────────────────────
# 共现矩阵计算
# ─────────────────────────────────────────────────────────

def compute_cooccurrence_matrix(responses):
    """
    从响应集合计算点对的共现频率矩阵。

    Args:
        responses: list of sets/lists, 每个元素是一组点 ID

    Returns:
        cooc: N×N numpy 矩阵, cooc[i,j] = 点i和j同时出现的次数
        all_points: list of 点 ID (原始 token)
        token_to_idx: dict, 点 ID → 矩阵索引
    """
    # 收集所有点
    all_points = set()
    for r in responses:
        all_points.update(r)
    all_points = sorted(all_points)
    token_to_idx = {t: i for i, t in enumerate(all_points)}
    N = len(all_points)

    cooc = np.zeros((N, N), dtype=np.float32)
    for r in tqdm.tqdm(responses, desc="计算共现矩阵"):
        idxs = [token_to_idx[t] for t in r]
        for i in idxs:
            for j in idxs:
                if i != j:
                    cooc[i, j] += 1.0
                    cooc[j, i] += 1.0

    # 归一化：除以每个点参与的总响应数（Jaccard-like）
    total_per_point = np.sum(cooc, axis=1, keepdims=True) + 1e-8
    cooc_normalized = cooc / total_per_point

    return cooc_normalized, all_points, token_to_idx


# ─────────────────────────────────────────────────────────
# GNN 增强的攻击
# ─────────────────────────────────────────────────────────

def gnn_range_attack(
    responses,
    model_or_path,
    threshold=0.5,
    device="cpu",
):
    """
    GNN 增强的 Range Attack: 用 GNN 预测边来替代泄漏放大 + 素数查找。

    Args:
        responses: list of sets, 采样后的查询响应
        model_or_path: 训练好的 EdgePredictionGNN 模型, 或模型权重文件路径
        threshold: 边预测阈值 (默认 0.5)
        device: 推理设备

    Returns:
        G: NetworkX 图 (节点索引 → 原始 token 通过 go_back 映射)
        edges_used: 使用的边数
        go_back: dict, 节点索引 → 原始点 ID
    """
    print("计算共现矩阵...")
    cooc_normalized, all_points, token_to_idx = compute_cooccurrence_matrix(responses)
    N = len(all_points)
    print(f"共 {N} 个点")

    # 加载模型
    model = _load_model(model_or_path, input_dim=N)

    # GNN 预测边
    print(f"GNN 预测边 (阈值={threshold})...")
    edges, edge_probs = predict_edges_from_cooccurrence(
        model, cooc_normalized, device=device, threshold=threshold
    )

    print(f"预测到 {len(edges)} 条边")

    # 构建 go_back: idx → 原始 token
    go_back = {i: all_points[i] for i in range(N)}

    # 构建图
    G = nx.Graph()
    # 先把所有节点添加进图（孤立点也保留）
    for i in range(N):
        G.add_node(tuple([all_points[i]]))

    for (i, j) in edges:
        G.add_edge(tuple([all_points[i]]), tuple([all_points[j]]))

    return G, len(edges), go_back


def _load_model(model_or_path, input_dim=None):
    """加载模型：如果是路径则从文件加载，否则直接返回模型对象。"""
    if isinstance(model_or_path, EdgePredictionGNN):
        return model_or_path

    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"模型文件不存在: {model_or_path}")

        checkpoint = torch.load(model_or_path, map_location="cpu", weights_only=False)

        # 从 checkpoint 读取模型配置
        input_dim = checkpoint.get("input_dim", input_dim)
        hidden_dim = checkpoint.get("hidden_dim", 64)
        emb_dim = checkpoint.get("emb_dim", 32)

        if input_dim is None:
            raise ValueError("无法确定 input_dim，请提供模型参数")

        model = EdgePredictionGNN(
            input_dim=input_dim, hidden_dim=hidden_dim, emb_dim=emb_dim
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    raise TypeError(f"不支持的模型类型: {type(model_or_path)}")


# ─────────────────────────────────────────────────────────
# 与原始方法对比评估
# ─────────────────────────────────────────────────────────

def get_correct_edges(map_to_original, points):
    """
    获取 ground truth 的邻接边（用于评估）。

    两个点在曼哈顿距离 <= 1 时视为相邻。
    与 attack.py 中的 get_correct_edges 逻辑一致。
    """
    opposite_map = {}
    for i in map_to_original:
        val = tuple(map_to_original[i])
        if val in opposite_map:
            opposite_map[val].append(i)
        else:
            opposite_map[val] = [i]

    correct_edges = set()
    for i in tqdm.tqdm(map_to_original, desc="计算 Ground Truth 边"):
        neighbors = []
        point = map_to_original[i]
        if len(point) == 3:
            x, y, z = point
            neighbors.extend([
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1),
            ])
        else:
            x, y = point
            neighbors.extend([
                (x+1, y), (x-1, y),
                (x, y+1), (x, y-1),
            ])
        for n in neighbors:
            if n in opposite_map:
                for n_token in opposite_map[n]:
                    correct_edges.add((tuple(point) if len(point) > 1 else point[0],
                                       tuple(n) if len(n) > 1 else n[0]))
    return correct_edges
