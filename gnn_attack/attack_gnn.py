"""
三阶段 GNN 攻击主入口 — 设计文档 §1-3

用法:
    # 一键全流程（训练 + 攻击）
    python attack_gnn.py --train --target-data cali_self

    # 仅推理（使用已有模型）
    python attack_gnn.py --phase1-model results/phase1_model.pth --target-data cali_self
"""
import argparse, sys, os, time, json
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_dataset
import process_database
from gnn_model import (
    EdgePredictionGNN, extract_node_features, build_cooc_message_graph,
    _get_correct_edges_at_scale, check_accuracy_with_edges,
)
from gnn_self_training import SelfTrainingLoop
from gnn_reconstruction import ForceDirectedLayout, CheckReconstructionFailure
from train_gnn import train_phase1


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = EdgePredictionGNN(
        feature_dim=ckpt.get("feature_dim", 16),
        hidden_dim=ckpt.get("hidden_dim", 64),
        emb_dim=ckpt.get("emb_dim", 32),
        num_mp_layers=ckpt.get("num_message_layers", 2),
    )
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device)


def main():
    parser = argparse.ArgumentParser(description="三阶段 GNN 攻击")
    # 模型
    parser.add_argument("--phase1-model", type=str, default=None,
                        help="Phase 1 预训练模型路径（不指定 --train 时必填）")
    parser.add_argument("--train", action="store_true",
                        help="自动执行 Phase 1 训练")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Phase 1 训练轮数（--train 时生效）")
    parser.add_argument("--train-samples", type=int, default=200,
                        help="Phase 1 合成样本数（--train 时生效）")
    # 数据
    parser.add_argument("--target-data", type=str, default="cali_self",
                        help="目标数据集")
    parser.add_argument("--p", type=float, default=100,
                        help="采样百分比")
    # 输出
    parser.add_argument("--output", type=str, default="results",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Phase 1: 训练（可选） ──
    if args.train:
        print(f"\n{'='*60}")
        print(f"三阶段 GNN 攻击: target={args.target_data}, p={args.p}%")
        print(f"  Phase 1 自动训练: epochs={args.train_epochs}, samples={args.train_samples}")
        print(f"{'='*60}\n")
        model_path = f"{args.output}/phase1_model.pth"
        train_phase1(
            epochs=args.train_epochs, samples=args.train_samples,
            save_path=model_path, device=args.device,
        )
        args.phase1_model = model_path
    else:
        if args.phase1_model is None:
            parser.error("必须指定 --phase1-model 或 --train")
        print(f"\n{'='*60}")
        print(f"三阶段 GNN 攻击: target={args.target_data}, p={args.p}%")
        print(f"  Phase 1 模型: {args.phase1_model}")
        print(f"{'='*60}\n")

    # ── 加载数据集 ──
    ds = load_dataset(args.target_data, 20, 20)
    points = ds["points"]
    map_to_original = ds["map_to_original"]
    N0, N1 = ds["N0"], ds["N1"]

    print("Phase 2 数据准备: 闭式公式计算共现矩阵...")
    all_pts = sorted(set(points))
    N = len(all_pts)

    # 获取按 all_pts 排序后的坐标数组 [N]
    xs = np.array([map_to_original[t][0] for t in all_pts], dtype=np.float32)
    ys = np.array([map_to_original[t][1] for t in all_pts], dtype=np.float32)

    # 闭式公式: C[i,j] = min(xi,xj) * min(yi,yj) * (N0-max(xi,xj)) * (N1-max(yi,yj))
    # 四个边界独立选择，乘积得到同时框住两点的矩形总数
    min_x = np.minimum(xs[:, None], xs[None, :])   # [N, N]
    max_x = np.maximum(xs[:, None], xs[None, :])
    min_y = np.minimum(ys[:, None], ys[None, :])
    max_y = np.maximum(ys[:, None], ys[None, :])

    C_target = min_x * min_y * (N0 - max_x) * (N1 - max_y)
    np.fill_diagonal(C_target, 0)   # 自共现置零
    C_target = C_target.astype(np.float32)

    # 采样率: Binomial(p) 模拟 → 期望 = 全量 × p/100, 方差 = c × p(1-p)/10000
    if args.p < 100:
        p_ratio = args.p / 100.0
        C_target = C_target * p_ratio
        # 二项相对噪声: σ_rel = sqrt((1-p_ratio) / (c * p_ratio))
        eps = 1e-10
        sigma_rel = np.sqrt((1.0 - p_ratio) / (np.abs(C_target) + eps))
        noise = np.random.randn(N, N).astype(np.float32) * sigma_rel * C_target
        C_target = np.clip(C_target + noise, 0, None)

    total = C_target.sum(axis=1, keepdims=True) + 1e-8
    C_target = C_target / total
    print(f"  共 {N} 个点, C_target [{N}×{N}], p={args.p}%")

    # ── Phase 2: 自训练 (§2) ──
    print(f"\n{'─'*40}\nPhase 2: 自训练\n{'─'*40}")
    model = load_model(args.phase1_model, args.device)
    t0 = time.time()
    model_fine, E_hat = SelfTrainingLoop(model, C_target, max_iter=10, device=args.device)
    t2 = time.time() - t0
    # 诊断: 推断边概率分布
    edge_probs = [p for _, _, p in E_hat]
    if edge_probs:
        q = np.percentile(edge_probs, [0, 25, 50, 75, 100])
        print(f"  Phase 2 完成: {len(E_hat)} 条推断边, prob=[min={q[0]:.3f} p25={q[1]:.3f} "
              f"med={q[2]:.3f} p75={q[3]:.3f} max={q[4]:.3f}], {t2:.1f}s")

    # ── Phase 3: 形状重建 (§3) ──
    print(f"\n{'─'*40}\nPhase 3: 形状重建\n{'─'*40}")
    t0 = time.time()
    pos = ForceDirectedLayout(E_hat, N)
    t3 = time.time() - t0
    failed, reason = CheckReconstructionFailure(pos, E_hat)
    if failed:
        print(f"  ⚠ 重建警告: {reason}")
    print(f"  Phase 3 完成: {t3:.1f}s")

    # ── 评估: Precision/Recall ──
    G = nx.Graph()
    id_to_token = {i: all_pts[i] for i in range(N)}
    for idx_i, idx_j, _ in E_hat:
        tok_i, tok_j = id_to_token[idx_i], id_to_token[idx_j]
        G.add_edge(tok_i, tok_j)
    precision, recall = check_accuracy_with_edges(G, map_to_original, points)
    print(f"\n{'─'*40}")
    print(f"评估: Precision={precision:.4f}  Recall={recall:.4f}  Edges={len(E_hat)}")

    # ── 输出 JSON ──
    result_file = f"{args.output}/result_{args.target_data}_p{args.p}.json"
    json.dump({
        "target": args.target_data, "p": args.p, "N": N,
        "edges": len(E_hat), "phase2_time": t2, "phase3_time": t3,
        "precision": precision, "recall": recall,
        "reconstruction_failed": failed, "reconstruction_reason": reason,
        "coordinates": pos.tolist()[:10],
    }, open(result_file, "w"), indent=2)
    print(f"结果已保存: {result_file}")

    # ── 可视化 ──
    plt.figure(figsize=(8, 8))
    plt.scatter(pos[:, 0], pos[:, 1], s=10)
    for i, j, _ in E_hat[:min(500, len(E_hat))]:
        plt.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 'gray', alpha=0.3, lw=0.5)
    plt.title(f"Reconstructed: {args.target_data}, p={args.p}%, "
              f"Precision={precision:.3f}, Recall={recall:.3f}, edges={len(E_hat)}")
    viz_file = f"{args.output}/recon_{args.target_data}_p{args.p}.png"
    plt.savefig(viz_file, dpi=100)
    print(f"可视化已保存: {viz_file}")


if __name__ == "__main__":
    main()
