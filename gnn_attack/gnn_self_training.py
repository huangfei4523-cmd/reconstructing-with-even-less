"""
Phase 2: 自训练模块 — 设计文档详细设计 §2

在目标加密数据的共现矩阵上，用预训练 GNN 的高置信度预测作为伪标签，
通过迭代微调使模型适应目标分布。
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from gnn_model import EdgePredictionGNN, extract_node_features, build_cooc_message_graph, extract_edge_features


def SelectPseudoLabels(prob_matrix, N):
    """
    §2.2: 从边概率矩阵筛选伪标签。
    Args: prob_matrix: [N,N] 上三角概率值; N: 节点数
    Returns: (pseudo_pos, pseudo_neg) — 边索引列表 [(i,j), ...]
    """
    # 取上三角
    triu_indices = np.triu_indices(N, k=1)
    probs = prob_matrix[triu_indices]
    sorted_idx = np.argsort(-probs)  # 降序
    K = 2 * N
    L = 10 * N

    # Top-K 作为正样本（带覆盖度检查）
    top_candidates = []
    for idx in sorted_idx:
        if len(top_candidates) >= K * 2:
            break
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        top_candidates.append((i, j))

    # 从候选中选 K 条，确保覆盖 ≥80% 节点
    pseudo_pos = []
    covered = set()
    for i, j in top_candidates:
        if len(pseudo_pos) >= K:
            break
        pseudo_pos.append((i, j))
        covered.add(i)
        covered.add(j)
    if len(covered) < 0.8 * N:
        # 补充未覆盖节点的最高分边
        for idx in sorted_idx:
            if len(pseudo_pos) >= K:
                break
            i, j = triu_indices[0][idx], triu_indices[1][idx]
            if i not in covered or j not in covered:
                pseudo_pos.append((i, j))
                covered.add(i)
                covered.add(j)

    # Bottom-L 作为负样本（避开已选中节点，防止混淆）
    pos_nodes = set()
    for i, j in pseudo_pos:
        pos_nodes.add(i)
        pos_nodes.add(j)

    pseudo_neg = []
    # 从最低概率中选，优先不共享节点的对
    candidates = list(reversed(sorted_idx))
    for idx in candidates:
        if len(pseudo_neg) >= L:
            break
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        if i not in pos_nodes and j not in pos_nodes:
            pseudo_neg.append((i, j))
    # 如果还不够，放宽约束
    for idx in candidates:
        if len(pseudo_neg) >= L:
            break
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        if (i, j) not in set(pseudo_pos):
            pseudo_neg.append((i, j))

    return pseudo_pos, pseudo_neg


def PerturbCooc(C, drop_pct=0.05):
    """§2.3: 对共现矩阵做随机扰动，模拟响应丢弃"""
    N = C.shape[0]
    rows, cols = np.where(C > 0)
    mask = np.random.random(len(rows)) > drop_pct
    C_pert = np.zeros_like(C)
    for i, j in zip(rows[mask], cols[mask]):
        C_pert[i, j] = C[i, j]
    return C_pert


def SelfTrainingLoop(model, C_target, max_iter=20, lr=0.0001, device="cpu"):
    """
    §2.3: 迭代自训练主循环。
    Args:
        model: 预训练 EdgePredictionGNN
        C_target: [N,N] 目标共现矩阵
        max_iter: 最大迭代轮数
        lr: 微调学习率 (≤ 预训练 lr/10)
    Returns: (model_fine, E_hat) — 微调后模型 + 推断边集合 [(i,j,prob)]
    """
    N = C_target.shape[0]
    # 诊断: C_target 统计
    row_sum = C_target.sum(axis=1)
    nonzero = (C_target > 0).sum(axis=1)
    print(f"  C_target: N={N}  row_strength={row_sum.mean():.4f}±{row_sum.std():.4f}  "
          f"nonzero_neighbors={nonzero.mean():.1f}±{nonzero.std():.1f}")
    model = model.to(device)
    optimizer = torch.optim.Adam([
        {"params": model.edge_predictor.parameters()},
        {"params": model.mp_layers[-1].parameters()},
    ], lr=lr)

    prev_pos = set()
    best_E_hat = []
    overlap_history = []

    for t in range(max_iter):
        t_start = time.time()
        # 1. 推理
        model.eval()
        node_feat = torch.FloatTensor(extract_node_features(C_target)).to(device)
        edge_idx, edge_w = build_cooc_message_graph(C_target)
        edge_idx_t = torch.LongTensor(edge_idx).to(device)
        edge_w_t = torch.FloatTensor(edge_w).to(device)

        with torch.no_grad():
            node_emb, _ = model(node_feat, edge_idx_t, None, edge_w_t)
            prob_matrix = model.predict_all_pairs(node_emb).cpu().numpy()

        # 2. 筛选伪标签
        pos, neg = SelectPseudoLabels(prob_matrix, N)

        # 诊断: 伪标签置信度
        pos_probs = [prob_matrix[i, j] for i, j in pos]
        neg_probs = [prob_matrix[i, j] for i, j in neg]
        pos_conf = np.mean(pos_probs) if pos_probs else 0
        neg_conf = np.mean(neg_probs) if neg_probs else 0
        all_probs = prob_matrix[np.triu_indices(N, k=1)]
        prob_entropy = -(all_probs * np.log(all_probs + 1e-10) + (1-all_probs) * np.log(1-all_probs + 1e-10)).mean()

        # 3. 收敛判定 (§2.4)
        curr_pos = set(pos)
        if prev_pos:
            overlap = len(curr_pos & prev_pos) / max(len(curr_pos), len(prev_pos), 1)
            overlap_history.append(overlap)
            if overlap >= 0.90:
                print(f"  Phase 2 收敛: iter={t}, overlap={overlap:.3f}")
                break
            # 发散检测
            if len(overlap_history) >= 3 and all(
                overlap_history[-i] < overlap_history[-i-1] for i in [1,2,3]
            ):
                print(f"  Phase 2 发散警告: iter={t}, overlap 连续下降")
                break
        prev_pos = curr_pos
        overlap_str = f"{overlap_history[-1]:.3f}" if overlap_history else "N/A"
        print(f"  iter {t}: pos={len(pos)}(conf={pos_conf:.3f})  neg={len(neg)}(conf={neg_conf:.3f})  "
              f"entropy={prob_entropy:.3f}  overlap={overlap_str}")

        # 4. 微调（§2.2 修复：完整共现图消息传递 + 伪标签边单独预测）
        model.train()
        pos_tensor = torch.tensor([[p[0], p[1]] for p in pos], dtype=torch.long, device=device)
        neg_tensor = torch.tensor([[n[0], n[1]] for n in neg], dtype=torch.long, device=device)

        # 用完整共现图做消息传递获得节点嵌入
        node_emb_train, _ = model(node_feat, edge_idx_t, None, edge_w_t)
        # 在伪标签边上单独拼接 src+dst 嵌入 → edge_predictor
        edge_feat_dim = model.edge_feat_dim
        zero_feat = torch.zeros(pos_tensor.size(0), edge_feat_dim, device=device)
        pos_input = torch.cat([node_emb_train[pos_tensor[:, 0]],
                               node_emb_train[pos_tensor[:, 1]], zero_feat], dim=-1)
        pos_logits = model.edge_predictor(pos_input).squeeze(-1)
        zero_feat_n = torch.zeros(neg_tensor.size(0), edge_feat_dim, device=device)
        neg_input = torch.cat([node_emb_train[neg_tensor[:, 0]],
                               node_emb_train[neg_tensor[:, 1]], zero_feat_n], dim=-1)
        neg_logits = model.edge_predictor(neg_input).squeeze(-1)

        # 一致性正则化（使用相同消息传递图，只替换节点特征）
        C_pert = PerturbCooc(C_target, 0.05)
        node_feat_p = torch.FloatTensor(extract_node_features(C_pert)).to(device)
        _, logits_p = model(node_feat_p, edge_idx_t, None, edge_w_t)
        _, logits_orig = model(node_feat, edge_idx_t, None, edge_w_t)
        consistency_loss = F.mse_loss(logits_orig, logits_p)

        bce_pos = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits))
        bce_neg = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits))
        loss = bce_pos + bce_neg + 0.1 * consistency_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"         loss={loss.item():.4f}  time={time.time()-t_start:.1f}s")

        best_E_hat = [(i, j, float(prob_matrix[i, j])) for i, j in curr_pos]

    return model, best_E_hat
