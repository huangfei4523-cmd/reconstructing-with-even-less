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

    print("Phase 2 数据准备: 流式计算目标共现矩阵...")
    responses = process_database.get_responses_no_vals(points, map_to_original, N0, N1)
    sample_size = max(1, int(len(responses) * args.p / 100))
    sampled = process_database.sample_uniform(responses, sample_size)

    # 流式构建共现矩阵：不存储全部响应集合，逐条处理
    all_pts_set = set()
    for min0, max0, min1, max1 in sampled:
        for p in points:
            if map_to_original[p][0] <= max0 and map_to_original[p][0] >= min0 \
               and map_to_original[p][1] <= max1 and map_to_original[p][1] >= min1:
                all_pts_set.add(p)
    all_pts = sorted(all_pts_set)
    token_to_idx = {t: i for i, t in enumerate(all_pts)}
    N = len(all_pts)
    C_target = np.zeros((N, N), dtype=np.float32)

    for min0, max0, min1, max1 in sampled:
        r = [p for p in points
             if map_to_original[p][0] <= max0 and map_to_original[p][0] >= min0
             and map_to_original[p][1] <= max1 and map_to_original[p][1] >= min1]
        if len(r) < 2:
            continue
        idxs = [token_to_idx[t] for t in r]
        for a in idxs:
            for b in idxs:
                if a != b:
                    C_target[a, b] += 1.0
                    C_target[b, a] += 1.0

    total = C_target.sum(axis=1, keepdims=True) + 1e-8
    C_target = C_target / total
    print(f"  共 {N} 个点, {len(sampled)} 条采样查询")

    # ── Phase 2: 自训练 (§2) ──
    print(f"\n{'─'*40}\nPhase 2: 自训练\n{'─'*40}")
    model = load_model(args.phase1_model, args.device)
    t0 = time.time()
    model_fine, E_hat = SelfTrainingLoop(model, C_target, max_iter=10, device=args.device)
    t2 = time.time() - t0
    print(f"  Phase 2 完成: {len(E_hat)} 条推断边, {t2:.1f}s")

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
