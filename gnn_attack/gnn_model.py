"""
GNN 模型：用图神经网络学习从共现矩阵到空间邻接的映射。

v1 架构（固定特征输入）:
  共现矩阵 → 统计特征提取（F=16 维，与 N 无关）
          → k-NN 稀疏图构建
          → GraphSAGE 风格消息传递 ×2
          → 边预测（拼接节点嵌入 + 边特征）

与 v0 的核心区别: 所有 nn.Linear 层输入维度固定，不受数据集点数 N 影响。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import tqdm


# ═══════════════════════════════════════════════════════════
# 1. 特征提取（固定维度，独立于 N）
# ═══════════════════════════════════════════════════════════

def extract_node_features(cooc_matrix):
    """
    从共现矩阵提取固定维度 F=16 的节点统计特征。

    Args:
        cooc_matrix: N×N float32 共现矩阵

    Returns:
        node_feat: [N, 16] float32 特征矩阵
    """
    N = cooc_matrix.shape[0]
    if N < 2:
        return np.zeros((N, 16), dtype=np.float32)

    feat = np.zeros((N, 16), dtype=np.float32)
    eps = 1e-8

    for i in range(N):
        vec = cooc_matrix[i]
        total = vec.sum() + eps
        nonzero = (vec > 0).sum()
        mean_v = vec.mean()
        std_v = vec.std()

        feat[i, 0] = mean_v / (N + eps)                      # F1: 平均共现度
        feat[i, 1] = vec.max() / (N + eps)                   # F2: 最大共现度
        feat[i, 2] = nonzero / (N + eps)                     # F3: 共现稀疏度
        feat[i, 3] = std_v / (N + eps)                       # F4: 标准差

        # F5: 共现偏度 (skewness of co-occurrence distribution)
        centered = vec - mean_v
        feat[i, 4] = (centered ** 3).mean() / (std_v ** 3 + eps)

        feat[i, 5] = total / (N * N + eps)                   # F6: 总强度归一化

        # F7: 共现峰度 (kurtosis of co-occurrence distribution)
        feat[i, 6] = (centered ** 4).mean() / (std_v ** 4 + eps)

        feat[i, 7] = 0.0                                     # F8: 聚类系数(计算开销大, 先留0)

        # F9-F10: 中心性近似
        norm_vec = vec / (vec.sum() + eps)
        feat[i, 8] = 1.0 / (nonzero + eps)                   # F9: 稀疏度倒数
        feat[i, 9] = total / (N + eps)                       # F10: 总强度

        # F11: 共现集中度
        k_top = min(3, N)
        topk = np.partition(vec, -k_top)[-k_top:] if k_top > 0 else vec
        feat[i, 10] = topk.sum() / (total + eps)

        # F12: 熵-like
        p = norm_vec[norm_vec > 0]
        feat[i, 11] = -(p * np.log(p + eps)).sum()           # F12: 自信息/熵

        # F13-F16: 保留位, 留 0

    return feat


def extract_edge_features(cooc_matrix, responses, edge_index):
    """
    为消息传递图的每条边计算 pairwise 统计特征。

    Args:
        cooc_matrix: N×N float32 共现矩阵
        responses: list of sets, 采样后的查询响应
        edge_index: [2, E] int64 边索引

    Returns:
        edge_feat: [E, 4] float32 边特征
    """
    N = cooc_matrix.shape[0]
    E = edge_index.shape[1]
    eps = 1e-8

    edge_feat = np.zeros((E, 4), dtype=np.float32)

    # 预处理: 每个点出现在哪些响应中
    point_responses = [set() for _ in range(N)]
    for ri, r in enumerate(responses):
        for t_idx in r:
            if isinstance(t_idx, int) and t_idx < N:
                point_responses[t_idx].add(ri)

    for e in range(E):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])

        # E1: 归一化共现计数
        cooc_ij = cooc_matrix[i, j]
        total_i = cooc_matrix[i].sum() + eps
        total_j = cooc_matrix[j].sum() + eps
        edge_feat[e, 0] = 2.0 * cooc_ij / (total_i + total_j + eps)

        # E2: Jaccard 系数
        if i < N and j < N and point_responses[i] and point_responses[j]:
            intersection = len(point_responses[i] & point_responses[j])
            union = len(point_responses[i] | point_responses[j])
            edge_feat[e, 1] = intersection / (union + eps)
        else:
            edge_feat[e, 1] = 0.0

        # E3: 余弦相似度
        vec_i = cooc_matrix[i]
        vec_j = cooc_matrix[j]
        dot = vec_i @ vec_j
        norm_i = np.linalg.norm(vec_i) + eps
        norm_j = np.linalg.norm(vec_j) + eps
        edge_feat[e, 2] = dot / (norm_i * norm_j)

        # E4: Adamic-Adar (简化)
        aa_sum = 0.0
        for k in range(N):
            if cooc_matrix[i, k] > 0 and cooc_matrix[j, k] > 0 and k != i and k != j:
                deg_k = (cooc_matrix[k] > 0).sum()
                if deg_k > 1:
                    aa_sum += 1.0 / np.log(deg_k + eps)
        edge_feat[e, 3] = aa_sum / (N + eps)

    return edge_feat


# ═══════════════════════════════════════════════════════════
# 2. 训练数据生成 v2
# ═══════════════════════════════════════════════════════════

def _generate_one_scene_2d_grid(N0, N1, max_points_per_cell, response_sampling_ratio):
    """生成一个 2D 规则网格场景的训练样本。"""
    map_to_original = {}
    for i in range(1, N0):
        for j in range(1, N1):
            repeats = int(1 + (max_points_per_cell - 1) * random.random())
            for _ in range(repeats):
                token = random.randrange(10000000)
                while token in map_to_original:
                    token = random.randrange(10000000)
                map_to_original[token] = (i, j)
    return _make_sample_from_points(map_to_original, (N0, N1), response_sampling_ratio, is_3d=False)


def _generate_one_scene_random_2d(n_points, x_range, y_range, response_sampling_ratio):
    """生成一个 2D 随机点云场景的训练样本。"""
    map_to_original = {}
    for _ in range(n_points):
        token = random.randrange(10000000)
        while token in map_to_original:
            token = random.randrange(10000000)
        x = random.randint(1, x_range)
        y = random.randint(1, y_range)
        map_to_original[token] = (x, y)
    return _make_sample_from_points(map_to_original, (x_range, y_range), response_sampling_ratio, is_3d=False)


def _generate_one_scene_random_3d(n_points, x_range, y_range, z_range, response_sampling_ratio):
    """生成一个 3D 随机点云场景的训练样本。"""
    map_to_original = {}
    for _ in range(n_points):
        token = random.randrange(10000000)
        while token in map_to_original:
            token = random.randrange(10000000)
        x = random.randint(1, x_range)
        y = random.randint(1, y_range)
        z = random.randint(1, z_range)
        map_to_original[token] = (x, y, z)
    return _make_sample_from_points(map_to_original, (x_range, y_range, z_range), response_sampling_ratio, is_3d=True)


def _make_sample_from_points(map_to_original, grid_bounds, response_sampling_ratio, is_3d=False):
    """从点集生成一个训练样本: 共现矩阵 → 节点特征 + 边标签。"""
    point_list = list(map_to_original.keys())
    N = len(point_list)
    token_to_idx = {t: i for i, t in enumerate(point_list)}

    if N < 4:
        return None

    # 生成范围查询（3D 场景采样优化，大网格避免枚举爆炸）
    responses_set = set()
    if is_3d:
        N0, N1, N2 = grid_bounds
        total_queries = (N0 * (N0 - 1) // 2) * (N1 * (N1 - 1) // 2) * (N2 * (N2 - 1) // 2)
        max_queries = 5000

        if total_queries > max_queries:
            # 随机采样
            for _ in range(max_queries):
                min0, max0 = sorted([random.randint(1, N0 - 1), random.randint(1, N0 - 1)])
                min1, max1 = sorted([random.randint(1, N1 - 1), random.randint(1, N1 - 1)])
                min2, max2 = sorted([random.randint(1, N2 - 1), random.randint(1, N2 - 1)])
                r = frozenset(
                    p for p in point_list
                    if min0 <= map_to_original[p][0] <= max0
                    and min1 <= map_to_original[p][1] <= max1
                    and min2 <= map_to_original[p][2] <= max2
                )
                if len(r) >= 2:
                    responses_set.add(r)
        else:
            for min0 in range(1, N0):
                for min1 in range(1, N1):
                    for min2 in range(1, N2):
                        for max0 in range(min0, N0):
                            for max1 in range(min1, N1):
                                for max2 in range(min2, N2):
                                    r = frozenset(
                                        p for p in point_list
                                        if min0 <= map_to_original[p][0] <= max0
                                        and min1 <= map_to_original[p][1] <= max1
                                        and min2 <= map_to_original[p][2] <= max2
                                    )
                                    if len(r) >= 2:
                                        responses_set.add(r)
    else:
        N0, N1 = grid_bounds
        for min0 in range(1, N0):
            for min1 in range(1, N1):
                for max0 in range(min0, N0):
                    for max1 in range(min1, N1):
                        r = frozenset(
                            p for p in point_list
                            if min0 <= map_to_original[p][0] <= max0
                            and min1 <= map_to_original[p][1] <= max1
                        )
                        if len(r) >= 2:
                            responses_set.add(r)

    responses = list(responses_set)
    sample_size = max(1, int(len(responses) * response_sampling_ratio))
    sampled = random.sample(responses, min(sample_size, len(responses)))

    # 共现矩阵
    cooc = np.zeros((N, N), dtype=np.float32)
    token_to_idx_map = token_to_idx
    for r in sampled:
        idxs = [token_to_idx_map[t] for t in r]
        for a in idxs:
            for b in idxs:
                if a != b:
                    cooc[a, b] += 1.0
                    cooc[b, a] += 1.0

    # 节点特征
    node_feat = extract_node_features(cooc)

    # Ground truth 邻接矩阵
    adj_gt = np.zeros((N, N), dtype=np.float32)
    for i, ti in enumerate(point_list):
        ci = map_to_original[ti]
        for j, tj in enumerate(point_list):
            if i >= j:
                continue
            cj = map_to_original[tj]
            dist = sum(abs(ci[d] - cj[d]) for d in range(len(ci)))
            if dist <= 1:
                adj_gt[i, j] = 1.0
                adj_gt[j, i] = 1.0

    return node_feat, adj_gt, sampled


def generate_training_data_v2(num_samples=500, configs=None):
    """
    生成多样化训练数据：支持 2D 网格、2D 随机点云、3D 随机点云。

    Args:
        num_samples: 样本总数
        configs: 场景配置列表，如 [{"type": "grid_2d", "ratio": 0.4, ...}, ...]

    Returns:
        node_features_list: list of [N, 16] 节点特征
        adjacency_list: list of [N, N] 邻接标签
        responses_list: list of list of sets, 每个样本的采样响应
    """
    if configs is None:
        # §1.2 Step 1: 参数空间网格 (N × 形状 × p)
        # N: 20~800, 形状: grid/random/circle/contour, p: 1%~50%
        # 总计 ~1500 样本覆盖 210 种配置
        ns = [20, 40, 80, 150, 300, 500, 800]
        ps = [0.01, 0.03, 0.05, 0.10, 0.20, 0.50]
        shapes_2d = ["grid_2d", "random_2d"]
        shapes_3d = ["random_3d"]
        
        configs = []
        for n in ns:
            for p in ps:
                # grid_2d: 规则栅格
                grid_sz = min(max(4, int(n ** 0.5)), 30)
                configs.append({"type": "grid_2d", "ratio": 1.0, "grid": (grid_sz, grid_sz),
                                "density": max(1, n // (grid_sz * grid_sz // 2)), "ratio_resp": p,
                                "N_override": n})
                # random_2d: 随机点云
                rng = max(10, int(n ** 0.6))
                configs.append({"type": "random_2d", "ratio": 1.0, "N_points": n,
                                "range": (rng, rng), "ratio_resp": p})
        
        # 3D 场景（占比 15%）
        for n in [40, 80, 150]:
            for p in [0.05, 0.20]:
                sz = max(3, int(n ** 0.33))
                configs.append({"type": "random_3d", "ratio": 0.25, "N_points": n,
                                "range": (sz, sz, sz), "ratio_resp": p})

    node_features_list = []
    adjacency_list = []
    responses_list = []

    pbar = tqdm.tqdm(total=num_samples, desc="生成训练数据 v2")

    # §1.2 Step 1: 轮询所有配置，每种配置生成 min(5, remaining) 个样本
    cfg_idx = 0
    while len(node_features_list) < num_samples:
        c = configs[cfg_idx]
        samples_per = min(5, num_samples - len(node_features_list))

        for _ in range(samples_per):
            if c["type"] == "grid_2d":
                g0, g1 = c["grid"]
                n_pts = c.get("N_override")
                if n_pts:
                    result = _generate_one_scene_random_2d(n_pts, g0 * 3, g1 * 3, c["ratio_resp"])
                else:
                    result = _generate_one_scene_2d_grid(g0, g1, c["density"], c["ratio_resp"])
            elif c["type"] == "random_2d":
                rx, ry = c["range"]
                result = _generate_one_scene_random_2d(c["N_points"], rx, ry, c["ratio_resp"])
            elif c["type"] == "random_3d":
                rx, ry, rz = c["range"]
                result = _generate_one_scene_random_3d(c["N_points"], rx, ry, rz, c["ratio_resp"])
            else:
                continue

            if result is not None:
                node_features_list.append(result[0])
                adjacency_list.append(result[1])
                responses_list.append(result[2])
                pbar.update(1)

        cfg_idx = (cfg_idx + 1) % len(configs)

    pbar.close()
    return node_features_list, adjacency_list, responses_list


# ═══════════════════════════════════════════════════════════
# 3. Dataset
# ═══════════════════════════════════════════════════════════

class CooccurrenceDataset(torch.utils.data.Dataset):
    """训练数据集：节点特征 + 邻接标签 + 采样响应。"""
    def __init__(self, node_features_list, adjacency_list, responses_list):
        self.node_features = [torch.FloatTensor(f) for f in node_features_list]
        self.adjacency = [torch.FloatTensor(a) for a in adjacency_list]
        self.responses = responses_list

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        # 将 frozenset 转为普通列表，以便 DataLoader 正确 collate
        resp = [list(r) for r in self.responses[idx]]
        return self.node_features[idx], self.adjacency[idx], resp


# ═══════════════════════════════════════════════════════════
# 4. 模型定义 (v1: 固定特征输入, GraphSAGE 风格)
# ═══════════════════════════════════════════════════════════

class EdgePredictionGNN(nn.Module):
    """
    v1 架构: 固定维度节点特征 + GraphSAGE 消息传递 + 边特征增强的边预测。

    关键: 所有 nn.Linear 输入维度与 N 无关。
    """

    def __init__(self, feature_dim=16, hidden_dim=64, emb_dim=32, edge_feat_dim=4, num_mp_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_mp_layers = num_mp_layers

        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )

        # 消息传递层 (GraphSAGE style)
        self.mp_layers = nn.ModuleList()
        self.mp_norms = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.mp_layers.append(nn.Linear(emb_dim * 2, emb_dim))
            self.mp_norms.append(nn.LayerNorm(emb_dim))

        # 边预测器 (拼接: emb_i + emb_j + edge_features)
        predictor_input = emb_dim * 2 + edge_feat_dim
        self.edge_predictor = nn.Sequential(
            nn.Linear(predictor_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_features, edge_index, edge_features=None, edge_weights=None):
        """
        Args:
            node_features: [N, feature_dim] 节点特征
            edge_index: [2, E] 稀疏边索引（消息传递图 = 所有共现对）
            edge_features: [E, edge_feat_dim] 边特征（可选，用于边预测）
            edge_weights: [E] 消息传递权重（= 归一化共现值）

        Returns:
            node_emb: [N, emb_dim] 节点嵌入
            edge_logits: [E] 各边的对数几率
        """
        N = node_features.shape[0]

        # 节点编码
        h = self.node_encoder(node_features)

        # 共现加权消息传递（设计文档 §1.3）
        for l_idx in range(self.num_mp_layers):
            row, col = edge_index[0], edge_index[1]

            # 加权聚合: agg_i = Σ_j w_ij * h_j / Σ_j w_ij
            msg = h[col]  # [E, emb_dim]
            if edge_weights is not None:
                w = edge_weights.unsqueeze(-1)  # [E, 1]
                msg = msg * w

            agg = torch.zeros_like(h)
            agg = agg.index_add(0, row, msg)
            deg = torch.zeros(N, 1, device=h.device)
            ones = torch.ones(msg.shape[0], 1, device=h.device)
            if edge_weights is not None:
                ones = ones * edge_weights.unsqueeze(-1)
            deg = deg.index_add(0, row, ones)
            deg = deg.clamp(min=1e-8)
            agg = agg / deg

            # GraphSAGE: concat(self, neighbors) → Linear
            h_cat = torch.cat([h, agg], dim=-1)
            h = self.mp_layers[l_idx](h_cat)
            h = self.mp_norms[l_idx](h)
            h = F.relu(h)

        node_emb = h

        # 边预测
        src_emb = node_emb[edge_index[0]]
        dst_emb = node_emb[edge_index[1]]
        pred_input = torch.cat([src_emb, dst_emb], dim=-1)

        if edge_features is not None:
            if edge_features.device != pred_input.device:
                edge_features = edge_features.to(pred_input.device)
            pred_input = torch.cat([pred_input, edge_features], dim=-1)
        else:
            zero_feat = torch.zeros(pred_input.shape[0], self.edge_feat_dim, device=pred_input.device)
            pred_input = torch.cat([pred_input, zero_feat], dim=-1)

        edge_logits = self.edge_predictor(pred_input).squeeze(-1)

        return node_emb, edge_logits

    def predict_all_pairs(self, node_emb, batch_size=2000):
        """
        对所有点对计算边概率（全连接推理）。
        对 N > 500 分批次计算以避免 OOM。

        Args:
            node_emb: [N, emb_dim]
            batch_size: 每批次处理的源节点数

        Returns:
            edge_probs: [N, N] 边概率矩阵
        """
        N = node_emb.shape[0]
        device = node_emb.device
        edge_probs = torch.zeros(N, N, device=device)

        for i_start in range(0, N, batch_size):
            i_end = min(i_start + batch_size, N)
            src_batch = node_emb[i_start:i_end]

            for j_start in range(0, N, batch_size):
                j_end = min(j_start + batch_size, N)
                dst_batch = node_emb[j_start:j_end]

                # 构造批次内所有点对
                ni = i_end - i_start
                nj = j_end - j_start
                src_exp = src_batch.unsqueeze(1).expand(-1, nj, -1).reshape(-1, self.emb_dim)
                dst_exp = dst_batch.unsqueeze(0).expand(ni, -1, -1).reshape(-1, self.emb_dim)

                pred_input = torch.cat([src_exp, dst_exp], dim=-1)
                # 无边缘特征，补零
                zero_feat = torch.zeros(pred_input.shape[0], self.edge_feat_dim, device=device)
                pred_input = torch.cat([pred_input, zero_feat], dim=-1)

                logits = self.edge_predictor(pred_input).squeeze(-1)
                probs = torch.sigmoid(logits).reshape(ni, nj)
                edge_probs[i_start:i_end, j_start:j_end] = probs

        # 对称化
        edge_probs = (edge_probs + edge_probs.T) / 2
        return edge_probs


# ═══════════════════════════════════════════════════════════
# 5. 训练逻辑（适配新数据格式）
# ═══════════════════════════════════════════════════════════

def train_gnn_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=0.001,
    device="cpu",
):
    """训练 GNN 边预测模型。使用 Focal Loss + 正样本加权。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            node_feat, adj_gt, responses = batch
            node_feat = node_feat.to(device)
            adj_gt = adj_gt.to(device)
            node_feat_2d = node_feat[0]     # [N, F] — 去掉 batch 维度
            adj_gt_2d = adj_gt[0]           # [N, N] — 去掉 batch 维度
            responses = responses[0]        # 去掉 batch 维度

            # §1.3: 构建共现加权消息传递图
            cooc_approx = node_feat_2d[:, :3].cpu().numpy()
            cooc_sim = cooc_approx @ cooc_approx.T
            edge_index, edge_weights = build_cooc_message_graph(cooc_sim)
            edge_index = torch.LongTensor(edge_index).to(device)
            edge_weights = torch.FloatTensor(edge_weights).to(device)

            if edge_index.shape[1] == 0:
                continue

            # 提取边特征
            edge_feat_np = extract_edge_features(cooc_sim, responses, edge_index.cpu().numpy())
            edge_feat = torch.FloatTensor(edge_feat_np).to(device)

            # 前向（带共现权重）
            _, edge_logits = model(node_feat_2d, edge_index, edge_feat, edge_weights)

            # 收集训练边的标签（只监督消息传递图中的边）
            edge_labels = adj_gt_2d[edge_index[0], edge_index[1]]

            pos_mask = edge_labels > 0.5
            if pos_mask.sum() == 0:
                continue

            pos_count = pos_mask.sum().float()
            neg_count = (~pos_mask).sum().float()
            pos_weight = neg_count / (pos_count + 1e-8)

            bce = F.binary_cross_entropy_with_logits(
                edge_logits, edge_labels,
                pos_weight=pos_weight.clone().detach(),
                reduction="none",
            )
            probs = torch.sigmoid(edge_logits)
            pt = torch.where(edge_labels > 0.5, probs, 1 - probs)
            focal_weight = (1 - pt) ** 2
            loss = (bce * focal_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # 验证
        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for node_feat, adj_gt, responses in val_loader:
                    node_feat = node_feat.to(device)
                    adj_gt = adj_gt.to(device)
                    node_feat_2d = node_feat[0]
                    adj_gt_2d = adj_gt[0]
                    responses = responses[0]

                    cooc_approx = node_feat_2d[:, :3].cpu().numpy()
                    cooc_sim = cooc_approx @ cooc_approx.T
                    edge_index, edge_weights = build_cooc_message_graph(cooc_sim)
                    edge_index = torch.LongTensor(edge_index).to(device)
                    edge_weights = torch.FloatTensor(edge_weights).to(device)

                    if edge_index.shape[1] == 0:
                        continue

                    edge_feat_np = extract_edge_features(cooc_sim, responses, edge_index.cpu().numpy())
                    edge_feat = torch.FloatTensor(edge_feat_np).to(device)

                    _, edge_logits = model(node_feat_2d, edge_index, edge_feat, edge_weights)
                    edge_labels = adj_gt_2d[edge_index[0], edge_index[1]]

                    pos_mask = edge_labels > 0.5
                    if pos_mask.sum() == 0:
                        continue
                    pos_count = pos_mask.sum().float()
                    neg_count = (~pos_mask).sum().float()
                    pos_weight = neg_count / (pos_count + 1e-8)
                    bce = F.binary_cross_entropy_with_logits(
                        edge_logits, edge_labels,
                        pos_weight=pos_weight.clone().detach(),
                        reduction="none",
                    )
                    probs = torch.sigmoid(edge_logits)
                    pt = torch.where(edge_labels > 0.5, probs, 1 - probs)
                    focal_weight = (1 - pt) ** 2
                    loss = (bce * focal_weight).mean()
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}  "
                    f"Train Loss: {avg_train_loss:.4f}  "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

        scheduler.step()

    return train_losses, val_losses


# ═══════════════════════════════════════════════════════════
# 6. 共现图消息传递构建（设计文档 §1.3）
# ═══════════════════════════════════════════════════════════

def build_cooc_message_graph(cooc_matrix):
    """
    基于共现矩阵构建加权消息传递图。
    边 = 所有 C[i,j] > 0 的点对。
    权重 = softmax(1 / log(1 + C[i,j]))。
    这与设计文档 §1.3 一致。

    Args:
        cooc_matrix: N×N 共现矩阵

    Returns:
        edge_index: [2, E] int64
        edge_weights: [E] float32, 归一化后用于消息传递的权重
    """
    N = cooc_matrix.shape[0]
    eps = 1e-8

    # 取所有共现对 (上三角，避免重复)
    rows, cols = np.where(np.triu(cooc_matrix > 0, k=1))
    if len(rows) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    # 对称化
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    # 共现权重: α_ij = 1 / log(1 + C[i,j])
    cooc_vals = cooc_matrix[rows, cols]
    w = 1.0 / np.log(1.0 + cooc_vals + eps)
    w = np.concatenate([w, w])  # 对称化

    # Softmax 归一化（每个目标节点对其入边做 softmax）
    w_normalized = np.zeros_like(w)
    for i in range(N):
        mask = edge_index[1] == i
        if mask.any():
            w_i = w[mask]
            w_i_exp = np.exp(w_i - w_i.max())  # 稳定 softmax
            w_normalized[mask] = w_i_exp / (w_i_exp.sum() + eps)

    return edge_index, w_normalized.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# 7. 推理
# ═══════════════════════════════════════════════════════════

def predict_edges_from_cooccurrence(model, cooc_matrix, responses=None, device="cpu", threshold=0.5):
    """
    用训练好的 GNN 从共现矩阵预测邻接边。

    Args:
        model: EdgePredictionGNN
        cooc_matrix: N×N numpy 共现矩阵
        responses: list of sets, 采样后的响应（用于提取边特征）
        device: 推理设备
        threshold: 边概率阈值

    Returns:
        edges: list of (i, j) tuples
        edge_probs: N×N 边概率矩阵
    """
    N = cooc_matrix.shape[0]
    if N < 2:
        return [], np.zeros((N, N))

    model = model.to(device)
    model.eval()

    # 提取节点特征
    node_feat = torch.FloatTensor(extract_node_features(cooc_matrix)).to(device)

    # §1.3: 构建共现加权消息传递图
    edge_index, edge_weights = build_cooc_message_graph(cooc_matrix)
    edge_index_t = torch.LongTensor(edge_index).to(device)
    edge_weights_t = torch.FloatTensor(edge_weights).to(device) if edge_weights.size > 0 else None

    # 提取边特征
    if responses is not None and edge_index.shape[1] > 0:
        edge_feat = extract_edge_features(cooc_matrix, responses, edge_index)
        edge_feat_t = torch.FloatTensor(edge_feat).to(device)
    else:
        edge_feat_t = None

    with torch.no_grad():
        node_emb, _ = model(node_feat, edge_index_t, edge_feat_t, edge_weights_t)
        edge_probs = model.predict_all_pairs(node_emb)
        edge_probs = edge_probs.cpu().numpy()

    # 提取边
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if edge_probs[i, j] >= threshold:
                edges.append((i, j))

    return edges, edge_probs


# ═══════════════════════════════════════════════════════════
# 8. 评估工具
# ═══════════════════════════════════════════════════════════

def _get_correct_edges_at_scale(points, dictionarry):
    """根据真实坐标计算正确答案边集合（曼哈顿距离 ≤1 的点对）。

    Args:
        points: 点 token 列表
        dictionarry: token → 坐标 (x,y) 或 (x,y,z) 的映射

    Returns:
        正确边集合 set of ((x1,y1), (x2,y2))
    """
    opposite_map = {}
    for i in dictionarry:
        val = tuple(dictionarry[i])
        if val in opposite_map:
            opposite_map[val].append(i)
        else:
            opposite_map[val] = [i]
    edges = set()
    for i in dictionarry:
        neighbors = []
        coord = dictionarry[i]
        if len(coord) == 3:
            x, y, z = coord
            neighbors = [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]
        else:
            x, y = coord
            neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        for n in neighbors:
            if n in opposite_map:
                for nt in opposite_map[n]:
                    edges.add((tuple(dictionarry[i]), tuple(dictionarry[nt])))
                    edges.add((tuple(dictionarry[nt]), tuple(dictionarry[i])))
    return edges


def check_accuracy_with_edges(G, dictionarry, points):
    """计算预测图的 Precision 和 Recall。

    Args:
        G: networkx 图
        dictionarry: token → 坐标的映射
        points: 点 token 列表

    Returns:
        precision, recall
    """
    import numpy as np
    correct_edges_set = set()
    incorrect = 0
    edges = list(G.edges())
    max_correct = _get_correct_edges_at_scale(points, dictionarry)

    for i in edges:
        correct_edge = False
        front_nodes = i[0]
        end_nodes = i[1]
        good = True
        point = dictionarry[front_nodes[0]]
        for f in list(front_nodes):
            if point != dictionarry[f]:
                good = False
        point = dictionarry[end_nodes[0]]
        for e in list(end_nodes):
            if point != dictionarry[e]:
                good = False
        if good:
            translated_node = (tuple(dictionarry[front_nodes[0]]), tuple(dictionarry[end_nodes[0]]))
            if np.linalg.norm(np.array(translated_node[0]) -
                               np.array(translated_node[1])) <= 1:
                correct_edge = True
        if correct_edge:
            tn = translated_node
            if tn not in correct_edges_set and (tn[1], tn[0]) not in correct_edges_set:
                correct_edges_set.add(tn)
                correct_edges_set.add((tn[1], tn[0]))
        else:
            incorrect += 1

    precision = len(correct_edges_set) / (len(correct_edges_set) + incorrect + 1e-10)
    recall = len(correct_edges_set) / (len(max_correct) + 1e-10)
    return precision, recall
