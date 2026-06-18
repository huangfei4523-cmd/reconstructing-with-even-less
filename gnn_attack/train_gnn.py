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
    generate_training_data,
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

    # 生成数据
    cooc_list, adj_list = generate_training_data(
        num_samples=args.samples,
        grid_size=tuple(args.grid),
        max_points_per_cell=2,
        response_sampling_ratio=args.ratio,
    )

    # 划分训练/验证
    n = len(cooc_list)
    n_val = max(1, int(n * args.val_split))
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_cooc = [cooc_list[i] for i in train_idx]
    train_adj = [adj_list[i] for i in train_idx]
    val_cooc = [cooc_list[i] for i in val_idx]
    val_adj = [adj_list[i] for i in val_idx]

    print(f"训练样本: {len(train_cooc)}, 验证样本: {len(val_cooc)}")

    # 构建 DataLoader
    train_dataset = CooccurrenceDataset(train_cooc, train_adj)
    val_dataset = CooccurrenceDataset(val_cooc, val_adj)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    # 创建模型
    input_dim = cooc_list[0].shape[-1]
    model = EdgePredictionGNN(
        input_dim=input_dim, hidden_dim=args.hidden, emb_dim=args.emb
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
            "input_dim": input_dim,
            "hidden_dim": args.hidden,
            "emb_dim": args.emb,
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
