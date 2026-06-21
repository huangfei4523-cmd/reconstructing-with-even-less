"""
Phase 1 训练脚本 — 设计文档 §1.2 Step 1-4

用法:
    python train_gnn.py --epochs 30 --samples 500 --save results/phase1_model.pth
"""
import argparse, os, torch, numpy as np
from gnn_model import (
    EdgePredictionGNN, generate_training_data_v2,
    CooccurrenceDataset, train_gnn_model, build_cooc_message_graph,
    extract_node_features, extract_edge_features, predict_edges_from_cooccurrence
)


def train_phase1(epochs=30, lr=0.001, hidden=64, emb=32, samples=500,
                 val_split=0.15, save_path="results/phase1_model.pth", device="cpu"):
    """Phase 1 预训练（可被外部导入调用）。

    Args:
        epochs: 训练轮数
        lr: 学习率
        hidden: 隐藏层维度
        emb: 嵌入维度
        samples: 合成样本数
        val_split: 验证集比例
        save_path: 模型保存路径
        device: 训练设备

    Returns:
        模型保存路径 (str)
    """
    print(f"Phase 1 预训练: epochs={epochs}, samples={samples}, device={device}")
    print(f"  §1.2 Step 1: 生成 {samples} 个训练样本 (参数空间网格)...")

    node_feat_list, adj_list, resp_list = generate_training_data_v2(
        num_samples=samples, configs=None
    )

    n = len(node_feat_list)
    n_val = max(1, int(n * val_split))
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_feat = [node_feat_list[i] for i in train_idx]
    train_adj = [adj_list[i] for i in train_idx]
    train_resp = [resp_list[i] for i in train_idx]
    val_feat = [node_feat_list[i] for i in val_idx]
    val_adj = [adj_list[i] for i in val_idx]
    val_resp = [resp_list[i] for i in val_idx]

    print(f"  训练样本: {len(train_feat)}, 验证样本: {len(val_feat)}")

    train_dataset = CooccurrenceDataset(train_feat, train_adj, train_resp)
    val_dataset = CooccurrenceDataset(val_feat, val_adj, val_resp)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = EdgePredictionGNN(feature_dim=16, hidden_dim=hidden, emb_dim=emb)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_gnn_model(
        model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
    )

    # §1.4 验证: 输出多阈值 P/R
    print("\n  §1.4 验证集评估:")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for node_feat, adj_gt, _ in val_loader:
            node_feat_2d = node_feat.to(device)[0]
            adj_gt_2d = adj_gt.to(device)[0]
            cooc_approx = node_feat_2d[:, :3].cpu().numpy()
            cooc_sim = cooc_approx @ cooc_approx.T
            edge_idx, edge_w = build_cooc_message_graph(cooc_sim)
            edge_idx = torch.LongTensor(edge_idx).to(device)
            edge_w = torch.FloatTensor(edge_w).to(device)
            _, logits = model(node_feat_2d, edge_idx, None, edge_w)
            probs = torch.sigmoid(logits)
            labels = adj_gt_2d[edge_idx[0], edge_idx[1]]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pred = (all_preds >= t).astype(float)
        tp = ((pred == 1) & (all_labels == 1)).sum()
        fp = ((pred == 1) & (all_labels == 0)).sum()
        fn = ((pred == 0) & (all_labels == 1)).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        print(f"    threshold={t}: P={prec:.3f} R={rec:.3f}")

    # 保存
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_dim": 16, "hidden_dim": hidden, "emb_dim": emb,
        "num_message_layers": 2, "train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_losses[-1] if val_losses else None,
    }, save_path)
    print(f"  模型已保存到: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Phase 1 预训练 GNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--emb", type=int, default=32)
    parser.add_argument("--save", type=str, default="results/phase1_model.pth")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--val-split", type=float, default=0.15)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_phase1(
        epochs=args.epochs, lr=args.lr, hidden=args.hidden, emb=args.emb,
        samples=args.samples, val_split=args.val_split,
        save_path=args.save, device=device,
    )


if __name__ == "__main__":
    main()
