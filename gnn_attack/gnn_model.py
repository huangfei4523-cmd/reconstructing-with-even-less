"""
GNN 模型：用图神经网络学习从共现矩阵到空间邻接的映射。

核心思想：
  原始攻击通过"集合交集→找大小为2的响应→建图"来推断邻接关系。
  这个 GNN 模型直接学习：给定点的共现特征，预测哪些点对是空间相邻的。
  它能利用所有响应大小中的模式（不局限于 size=2），在信息稀疏时可能更鲁棒。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import tqdm
from collections import defaultdict


# ─────────────────────────────────────────────────────────
# 训练数据生成
# ─────────────────────────────────────────────────────────

def generate_training_data(
    num_samples=500,
    grid_size=(10, 10),
    max_points_per_cell=2,
    response_sampling_ratio=0.05,
):
    """
    生成训练数据：在小网格上模拟攻击场景。
    
    返回:
        cooccurrence_matrices: list of N×N 共现矩阵
        adjacency_matrices: list of N×N 邻接矩阵 (ground truth)
    """
    cooccurrence_list = []
    adjacency_list = []

    for _ in tqdm.tqdm(range(num_samples), desc="生成训练数据"):
        # 1. 随机生成点集
        points = {}  # token -> (x, y)
        map_to_original = {}
        N0, N1 = grid_size

        for i in range(1, N0):
            for j in range(1, N1):
                repeats = int(1 + (max_points_per_cell - 1) * random.random())
                for _ in range(repeats):
                    token = random.randrange(10000000)
                    while token in map_to_original:
                        token = random.randrange(10000000)
                    map_to_original[token] = (i, j)
                    points[token] = (i, j)

        point_list = list(points.keys())
        N = len(point_list)
        if N < 4:
            continue

        # 2. 生成全部范围查询响应
        responses = set()
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
                            responses.add(r)

        responses = list(responses)

        # 3. 采样少量响应 (模拟 "Even Less")
        sample_size = max(1, int(len(responses) * response_sampling_ratio))
        sampled = random.sample(responses, min(sample_size, len(responses)))

        # 4. 计算共现矩阵
        token_to_idx = {t: i for i, t in enumerate(point_list)}
        cooc = np.zeros((N, N), dtype=np.float32)
        for r in sampled:
            idxs = [token_to_idx[t] for t in r]
            for i in idxs:
                for j in idxs:
                    if i != j:
                        cooc[i, j] += 1.0
                        cooc[j, i] += 1.0

        # 5. Ground truth 邻接矩阵: 曼哈顿距离 <= 1 的点对
        adj_gt = np.zeros((N, N), dtype=np.float32)
        for i, ti in enumerate(point_list):
            xi, yi = map_to_original[ti]
            for j, tj in enumerate(point_list):
                if i >= j:
                    continue
                xj, yj = map_to_original[tj]
                if abs(xi - xj) + abs(yi - yj) <= 1:
                    adj_gt[i, j] = 1.0
                    adj_gt[j, i] = 1.0

        cooccurrence_list.append(cooc)
        adjacency_list.append(adj_gt)

    return cooccurrence_list, adjacency_list


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class CooccurrenceDataset(torch.utils.data.Dataset):
    def __init__(self, cooccurrence_list, adjacency_list):
        self.cooccurrence = [torch.FloatTensor(m) for m in cooccurrence_list]
        self.adjacency = [torch.FloatTensor(a) for a in adjacency_list]

    def __len__(self):
        return len(self.cooccurrence)

    def __getitem__(self, idx):
        return self.cooccurrence[idx], self.adjacency[idx]


# ─────────────────────────────────────────────────────────
# 模型定义
# ─────────────────────────────────────────────────────────

class EdgePredictionGNN(nn.Module):
    """
    双消息传递层 GNN + 边预测头。

    设计思路:
      - 每个点的初始特征是它与其他点的共现频率向量
      - GCN 风格的消息传递让相邻节点交换信息
      - 学到的节点嵌入通过 MLP 预测任意两点是否相邻
    """

    def __init__(self, input_dim, hidden_dim=64, emb_dim=32):
        super().__init__()

        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        # 消息传递层 (GCN-style)
        self.mp1 = nn.Linear(emb_dim, emb_dim)
        self.mp2 = nn.Linear(emb_dim, emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        # 边预测器
        self.edge_predictor = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, adj):
        """
        Args:
            x: 节点特征 [N, input_dim]
            adj: 归一化邻接矩阵 [N, N] 用于消息传递

        Returns:
            node_emb: [N, emb_dim]
            edge_logits: [N, N] 未归一化的边预测分数
        """
        N = x.shape[0]

        # 编码
        h = self.node_encoder(x)

        # 消息传递 1
        h = self.mp1(adj @ h)
        h = self.norm1(h)
        h = F.relu(h)

        # 消息传递 2
        h = self.mp2(adj @ h)
        h = self.norm2(h)
        h = F.relu(h)

        node_emb = h

        # 边预测：对所有 (i,j) 对
        emb_i = node_emb.unsqueeze(1).expand(-1, N, -1)
        emb_j = node_emb.unsqueeze(0).expand(N, -1, -1)
        edge_feat = torch.cat([emb_i, emb_j], dim=-1)
        edge_logits = self.edge_predictor(edge_feat).squeeze(-1)

        # 对称化
        edge_logits = (edge_logits + edge_logits.T) / 2

        return node_emb, edge_logits


# ─────────────────────────────────────────────────────────
# 训练逻辑
# ─────────────────────────────────────────────────────────

def train_gnn_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=0.001,
    device="cpu",
):
    """
    训练 GNN 边预测模型。

    Loss 设计:
      - 正负样本严重不平衡（大多数点对不相邻）
      - 使用 Focal Loss 缓解类别不平衡
      - 对正样本加权
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # ── 训练 ──
        model.train()
        total_loss = 0
        num_batches = 0

        for cooc, adj_gt in train_loader:
            cooc = cooc.to(device)
            adj_gt = adj_gt.to(device)
            N = cooc.shape[1]

            # 构建消息传递图：k-NN 基于共现相似度
            knn_adj = _build_knn_graph(cooc[0].cpu().numpy(), k=10)
            knn_adj = torch.FloatTensor(knn_adj).to(device)

            # 前向
            _, edge_logits = model(cooc, knn_adj)

            # 边数加权 Focal Loss
            pos_mask = adj_gt > 0.5
            neg_mask = ~pos_mask

            pos_count = pos_mask.sum().float()
            neg_count = neg_mask.sum().float()

            if pos_count == 0:
                continue

            # 正样本加权
            pos_weight = neg_count / (pos_count + 1e-8)

            bce = F.binary_cross_entropy_with_logits(
                edge_logits,
                adj_gt,
                pos_weight=torch.tensor(pos_weight).to(device),
                reduction="none",
            )

            # Focal 风格: 对易分样本降权
            probs = torch.sigmoid(edge_logits)
            pt = torch.where(adj_gt > 0.5, probs, 1 - probs)
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

        # ── 验证 ──
        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for cooc, adj_gt in val_loader:
                    cooc = cooc.to(device)
                    adj_gt = adj_gt.to(device)
                    N = cooc.shape[1]
                    knn_adj = _build_knn_graph(cooc[0].cpu().numpy(), k=10)
                    knn_adj = torch.FloatTensor(knn_adj).to(device)
                    _, edge_logits = model(cooc, knn_adj)

                    pos_mask = adj_gt > 0.5
                    neg_mask = ~pos_mask
                    if pos_mask.sum() == 0:
                        continue
                    pos_weight = neg_mask.sum() / (pos_mask.sum() + 1e-8)
                    bce = F.binary_cross_entropy_with_logits(
                        edge_logits,
                        adj_gt,
                        pos_weight=torch.tensor(pos_weight).to(device),
                        reduction="none",
                    )
                    probs = torch.sigmoid(edge_logits)
                    pt = torch.where(adj_gt > 0.5, probs, 1 - probs)
                    focal_weight = (1 - pt) ** 2
                    loss = (bce * focal_weight).mean()
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}  "
                    f"Train Loss: {avg_train_loss:.4f}  "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

    return train_losses, val_losses


def _build_knn_graph(cooc_matrix, k=10):
    """
    基于共现相似度构建 k-NN 图用于消息传递。
    使用余弦相似度找到每个点的 k 个最近邻。
    """
    N = cooc_matrix.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        vec_i = cooc_matrix[i]
        # 余弦相似度
        sim = vec_i @ cooc_matrix.T
        norms = np.linalg.norm(cooc_matrix, axis=1)
        norms[norms == 0] = 1
        sim = sim / (np.linalg.norm(vec_i) + 1e-8) / (norms + 1e-8)
        sim[i] = -np.inf  # 排除自身
        top_k = np.argsort(-sim)[: min(k, N)]
        adj[i, top_k] = 1.0

    # 对称化
    adj = np.maximum(adj, adj.T)

    # 归一化 (GCN style)
    deg = adj.sum(axis=1, keepdims=True) + 1e-8
    adj_norm = adj / deg

    return adj_norm


# ─────────────────────────────────────────────────────────
# 推理: 从共现矩阵预测边
# ─────────────────────────────────────────────────────────

def predict_edges_from_cooccurrence(model, cooc_matrix, device="cpu", threshold=0.5):
    """
    用训练好的 GNN 从共现矩阵预测邻接边。

    Args:
        model: 训练好的 EdgePredictionGNN
        cooc_matrix: N×N numpy 共现矩阵
        threshold: 边概率阈值 (默认 0.5)

    Returns:
        edges: list of (i, j) tuples
        edge_probs: N×N 边概率矩阵
    """
    N = cooc_matrix.shape[0]
    if N < 2:
        return [], np.zeros((N, N))

    model = model.to(device)
    model.eval()

    x = torch.FloatTensor(cooc_matrix).unsqueeze(0).to(device)

    # 构建 k-NN 消息传递图
    knn_adj = _build_knn_graph(cooc_matrix, k=min(10, N - 1))
    knn_adj = torch.FloatTensor(knn_adj).unsqueeze(0).to(device)

    with torch.no_grad():
        _, edge_logits = model(x, knn_adj)
        edge_probs = torch.sigmoid(edge_logits).squeeze(0).cpu().numpy()

    # 提取边
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if edge_probs[i, j] >= threshold:
                edges.append((i, j))

    return edges, edge_probs
