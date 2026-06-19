"""
训练 GNN 模型脚本。

用法:
    python train_gnn.py                      # 默认参数训练
    python train_gnn.py --epochs 100 --lr 0.001 --save gnns/grid_model.pt
"""

import argparse
import os
import sys
import numpy as np
import torch

from gnn_model import (
    EdgePredictionGNN,
    generate_training_data_v2,
    CooccurrenceDataset,
    train_gnn_model,
)


def main():
    parser = argparse.ArgumentParser(description="训练 GNN 边预测模型")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--hidden", type=int, default=64, help="隐藏层维度")
    parser.add_argument("--emb", type=int, default=32, help="节点嵌入维度")
    parser.add_argument(
        "--save", type=str, default="gnn_model.pth", help="模型保存路径"
    )
    parser.add_argument(
        "--samples", type=int, default=500, help="训练数据样本数"
    )
    parser.add_argument(
        "--grid", type=int, nargs=2, default=[10, 10], help="网格大小 N0 N1"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.05, help="响应采样比例"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="验证集比例"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"生成 {args.samples} 个训练样本...")

    # 生成数据（v2: 多场景混合）
    node_feat_list, adj_list, resp_list = generate_training_data_v2(
        num_samples=args.samples,
        configs=None,  # 使用默认多场景配置
    )

    # 划分训练/验证
    n = len(node_feat_list)
    n_val = max(1, int(n * args.val_split))
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_feat = [node_feat_list[i] for i in train_idx]
    train_adj = [adj_list[i] for i in train_idx]
    train_resp = [resp_list[i] for i in train_idx]
    val_feat = [node_feat_list[i] for i in val_idx]
    val_adj = [adj_list[i] for i in val_idx]
    val_resp = [resp_list[i] for i in val_idx]

    print(f"训练样本: {len(train_feat)}, 验证样本: {len(val_feat)}")

    # 构建 DataLoader
    train_dataset = CooccurrenceDataset(train_feat, train_adj, train_resp)
    val_dataset = CooccurrenceDataset(val_feat, val_adj, val_resp)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    # 创建模型（固定 feature_dim=16，与 N 无关）
    model = EdgePredictionGNN(
        feature_dim=16, hidden_dim=args.hidden, emb_dim=args.emb
    )
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    train_losses, val_losses = train_gnn_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # 保存模型
    save_dir = os.path.dirname(args.save) if os.path.dirname(args.save) else "."
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": 16,
            "hidden_dim": args.hidden,
            "emb_dim": args.emb,
            "num_message_layers": 2,
            "train_loss": train_losses[-1] if train_losses else None,
            "val_loss": val_losses[-1] if val_losses else None,
        },
        args.save,
    )
    print(f"模型已保存到: {args.save}")
    print(f"最终 Train Loss: {train_losses[-1]:.4f}" + 
          (f"  Val Loss: {val_losses[-1]:.4f}" if val_losses else ""))


if __name__ == "__main__":
    main()
