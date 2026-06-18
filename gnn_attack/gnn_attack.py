"""
GNN 增强的 Reconstructing with Even Less 攻击入口脚本。

用法:
    # 先训练模型
    python train_gnn.py --epochs 50 --save gnn_model.pth

    # 用 GNN 增强的攻击运行
    python gnn_attack.py -points=cali_50 -dist=uniform -p=1 --model gnn_model.pth

    # 与原始方法对比
    python gnn_attack.py -points=cali_50 -dist=uniform -p=1 --model gnn_model.pth --baseline

与原版的区别:
  - 泄漏放大步骤由 GNN 替代
  - GNN 从共现模式中学习推断邻接关系
  - 支持 --baseline 开关同时运行原始方法做对比
  - 自动保存评估结果到 JSON
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import argparse
import networkx as nx
import time
import numpy as np
import json
import os
import sys

import process_database
import gnn_range_attack as gnn_ra
import range_attack  # 原始方法, 用于 baseline 对比
from data_loader import load_dataset


# ─────────────────────────────────────────────────────────
# 精度评估
# ─────────────────────────────────────────────────────────

def check_accuracy_with_edges(G, dictionarry, points):
    """与原始 attack.py 相同的评估逻辑。"""
    correct = 0
    incorrect = 0
    edges = list(G.edges())
    correct_edges_set = set()
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
            translated_node = (tuple(dictionarry[f]), tuple(dictionarry[e]))
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


def _get_correct_edges_at_scale(points, dictionarry):
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
                    edges.add((tuple(dictionarry[i]), tuple(coord)))
                    edges.add((tuple(coord), tuple(dictionarry[i])))
    return edges


def plot_and_save(pos, title, filename):
    plt.figure(figsize=(8, 8))
    vals = list(pos.values())
    if vals and len(vals[0]) == 3:
        ax = plt.figure().add_subplot(projection="3d")
        X, Y, Z = zip(*vals)
        ax.scatter(X, Y, Z, s=5)
        ax.set_title(title)
    else:
        X, Y = zip(*[v[:2] for v in vals])
        plt.scatter(X, Y, s=5)
        plt.title(title)
        plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  图已保存: {filename}")


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GNN 增强的 Dense Attacks!")
    parser.add_argument("-points", type=str, default="small_grid",
                        help="数据集: cali_50, grid, dg, crg, nh, boat")
    parser.add_argument("-p", type=float, default=100, help="查询采样百分比")
    parser.add_argument("-dist", type=str, default="uniform",
                        help="采样分布: beta, gaussian, uniform")
    parser.add_argument("-N0", type=int, default=20, help="N0")
    parser.add_argument("-N1", type=int, default=20, help="N1")
    parser.add_argument("--model", type=str, default="gnn_model.pth",
                        help="GNN 模型权重文件路径")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="GNN 边预测阈值")
    parser.add_argument("--baseline", action="store_true",
                        help="同时运行原始方法做对比")
    parser.add_argument("--compare-only", action="store_true",
                        help="仅做对比，不保存图片")
    parser.add_argument("--device", type=str, default="cpu",
                        help="推理设备 (cpu 或 cuda)")
    parser.add_argument("--output", type=str, default="results",
                        help="输出目录")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GNN 增强攻击: points={args.points}, p={args.p}%, dist={args.dist}")
    print(f"{'='*60}\n")

    dataset_config = load_dataset(args.points, args.N0, args.N1)
    if dataset_config["points"] is None:
        print(f"数据集 {args.points} 加载失败")
        sys.exit(1)

    points = dataset_config["points"]
    map_to_original = dataset_config["map_to_original"]
    N0 = dataset_config["N0"]
    N1 = dataset_config["N1"]
    is_3d = dataset_config.get("is_3d", False)
    N2 = dataset_config.get("N2", None)

    # ── 生成查询响应 ──
    print("生成查询响应...")
    if is_3d:
        responses = process_database.get_responses_no_vals_3D(points, map_to_original, N0, N1, N2)
    else:
        responses = process_database.get_responses_no_vals(points, map_to_original, N0, N1)

    # ── 采样 ──
    print("采样响应...")
    sample_size = max(1, int(len(responses) * args.p / 100.0))
    if args.dist == "uniform":
        sampled = process_database.sample_uniform(responses, sample_size)
    elif args.dist == "beta":
        sampled = process_database.sample_beta(responses, sample_size)
    elif args.dist == "gaussian":
        sampled = process_database.sample_gaussian(responses, sample_size)
    else:
        print(f"未知采样分布: {args.dist}")
        sys.exit(1)

    # ── 翻译为实际响应 ──
    if is_3d:
        new_responses, unique_rs = process_database.get_actual_resps_after_sampling_3D(
            sampled, points, map_to_original
        )
    else:
        new_responses, unique_rs = process_database.get_actual_resps_after_sampling(
            sampled, points, map_to_original
        )

    results = {
        "config": {
            "points": args.points,
            "p": args.p,
            "dist": args.dist,
            "threshold": args.threshold,
            "model": args.model,
        },
        "gnn": {},
    }

    # ── GNN 攻击 ──
    print(f"\n{'─'*40}")
    print("运行 GNN 增强攻击...")
    print(f"{'─'*40}")
    start = time.time()
    G_gnn, used_gnn, go_back = gnn_ra.gnn_range_attack(
        new_responses, args.model, threshold=args.threshold, device=args.device
    )
    end = time.time()
    gnn_time = end - start

    precision_gnn, recall_gnn = check_accuracy_with_edges(G_gnn, map_to_original, points)
    print(f"GNN 攻击: Precision={precision_gnn:.4f}, Recall={recall_gnn:.4f}, Time={gnn_time:.2f}s")

    results["gnn"] = {
        "precision": float(precision_gnn),
        "recall": float(recall_gnn),
        "time": gnn_time,
        "edges_used": used_gnn,
    }

    # ── GNN 可视化 ──
    if not args.compare_only:
        os.makedirs(args.output, exist_ok=True)
        dim = 3 if is_3d else 2
        pos_gnn = nx.kamada_kawai_layout(G_gnn, dim=dim)
        plot_and_save(
            pos_gnn,
            f"GNN Attack: {args.points}, p={args.p}% (P={precision_gnn:.3f}, R={recall_gnn:.3f})",
            os.path.join(args.output, f"gnn_{args.points}_p{args.p}.png"),
        )

    # ── Baseline: 原始方法对比 ──
    if args.baseline:
        print(f"\n{'─'*40}")
        print("运行原始攻击方法 (Baseline)...")
        print(f"{'─'*40}")
        start = time.time()
        G_base, used_base = range_attack.general(new_responses)
        end = time.time()
        base_time = end - start

        precision_base, recall_base = check_accuracy_with_edges(G_base, map_to_original, points)
        print(f"原始攻击: Precision={precision_base:.4f}, Recall={recall_base:.4f}, Time={base_time:.2f}s")

        results["baseline"] = {
            "precision": float(precision_base),
            "recall": float(recall_base),
            "time": base_time,
            "edges_used": used_base,
        }

        # 可视化
        if not args.compare_only:
            pos_base = nx.kamada_kawai_layout(G_base, dim=dim)
            plot_and_save(
                pos_base,
                f"Original Attack: {args.points}, p={args.p}% (P={precision_base:.3f}, R={recall_base:.3f})",
                os.path.join(args.output, f"original_{args.points}_p{args.p}.png"),
            )

    # ── 保存结果 ──
    os.makedirs(args.output, exist_ok=True)
    result_file = os.path.join(
        args.output,
        f"results_{args.points}_p{args.p}_{args.dist}.json",
    )
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {result_file}")

    # ── 最终摘要 ──
    print(f"\n{'='*60}")
    print(f"摘要: {args.points}, p={args.p}%, dist={args.dist}")
    print(f"  GNN:     P={precision_gnn:.4f}, R={recall_gnn:.4f}, T={gnn_time:.1f}s")
    if args.baseline:
        print(f"  Original: P={precision_base:.4f}, R={recall_base:.4f}, T={base_time:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
