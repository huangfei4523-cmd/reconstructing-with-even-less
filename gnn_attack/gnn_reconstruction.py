"""
Phase 3: 形状重建模块 — 设计文档详细设计 §3

从推断的边集合通过力导向布局恢复 2D 点分布。
"""
import numpy as np
import networkx as nx


def ForceDirectedLayout(E_hat, N, anchors=None, max_iter=200, eps=1e-4):
    """
    §3.2: 弹簧-斥力模型。
    Args:
        E_hat: [(i,j,prob), ...]
        N: 节点数
        anchors: 可选 {(k, (x,y)), ...} 锚点坐标
    Returns: pos[N,2] — 推测的 2D 坐标
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i, j, prob in E_hat:
        G.add_edge(i, j, weight=prob)

    if G.number_of_edges() == 0:
        # 无边的退化情况：随机分布
        return np.random.rand(N, 2)

    # 初始位置
    if anchors:
        pos_init = {}
        for k, (x, y) in anchors.items():
            pos_init[k] = np.array([x, y])
        for i in range(N):
            if i not in pos_init:
                pos_init[i] = np.random.randn(2) * 0.1
        fixed_list = list(anchors.keys())
    else:
        pos_init = {i: np.array([np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)]) for i in range(N)}
        fixed_list = None

    kwargs = dict(weight='weight', k=1.0/np.sqrt(N), iterations=max_iter, threshold=eps)
    if fixed_list:
        kwargs["fixed"] = fixed_list
    pos = nx.spring_layout(G, pos=pos_init, **kwargs)

    result = np.array([pos[i] for i in range(N)])
    return result


def ProcrustesAlign(pred_pos, anchors):
    """
    §3.3: Procrustes 对齐。
    Args:
        pred_pos: [N,2] 推测坐标
        anchors: {(k, (true_x, true_y)), ...}
    Returns: aligned_pos[N,2], (rotation, scale, translation)
    """
    anchor_ids = list(anchors.keys())
    A = np.array([pred_pos[k] for k in anchor_ids])
    B = np.array([anchors[k] for k in anchor_ids])

    # Procrustes: find optimal R, s, t minimizing ||s*A*R + t - B||
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    A_centered = A - A_mean
    B_centered = B - B_mean

    U, _, Vt = np.linalg.svd(B_centered.T @ A_centered)
    R = (U @ Vt).T
    s = np.sum(B_centered * (A_centered @ R.T)) / np.sum(A_centered ** 2)
    t = B_mean - s * A_mean @ R.T

    aligned = s * pred_pos @ R.T + t
    return aligned, (R, s, t)


def HausdorffDist(A, B):
    """§3.3: Hausdorff 距离"""
    from scipy.spatial import cKDTree
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)
    d_AB = tree_B.query(A)[0].max()
    d_BA = tree_A.query(B)[0].max()
    return max(d_AB, d_BA)


def CheckReconstructionFailure(pos, E_hat):
    """
    §3.4: 重建失败检测。
    Returns: (is_failed, reason)
    """
    N = pos.shape[0]
    n_edges = len(E_hat)

    if n_edges < N / 2:
        return True, f"边数不足: {n_edges} < N/2={N/2}"

    var = pos.var(axis=0).sum()
    if var < 1e-6:
        return True, f"布局方差过小: {var}"

    # 连通分量检查
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(i, j) for i, j, _ in E_hat])
    n_components = nx.number_connected_components(G)
    if n_components > N / 4:
        return True, f"连通分量过多: {n_components} > N/4"

    return False, "OK"
