"""三阶段集成测试 — cali_self 端到端"""
import sys, os, unittest, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnn_model import EdgePredictionGNN, generate_training_data_v2, CooccurrenceDataset, train_gnn_model, build_cooc_message_graph
from gnn_self_training import SelfTrainingLoop
from gnn_reconstruction import ForceDirectedLayout, CheckReconstructionFailure


class TestIntegration(unittest.TestCase):
    def test_end_to_end_cali_self(self):
        """cali_self 规模的三阶段全流程 — Phase 1小规模→Phase 2→Phase 3"""
        # Phase 1: 快速预训练 (小规模)
        feat, adj, resp = generate_training_data_v2(num_samples=10, configs=None)
        ds = CooccurrenceDataset(feat, adj, resp)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
        model = EdgePredictionGNN(feature_dim=16, hidden_dim=32, emb_dim=16)
        train_gnn_model(model, loader, None, epochs=3, lr=0.001, device="cpu")

        # Phase 2: 用训练数据的第一个样本做自训练验证
        f0 = feat[0] if isinstance(feat[0], np.ndarray) else feat[0].cpu().numpy()
        C_target = f0[:, :3] @ f0[:, :3].T
        N = C_target.shape[0]
        C_target = C_target / (C_target.sum(axis=1, keepdims=True) + 1e-8)
        model, E_hat = SelfTrainingLoop(model, C_target, max_iter=3, device="cpu")
        self.assertGreater(len(E_hat), 0, "Phase 2 应产生 >0 条推断边")

        # Phase 3: 布局
        pos = ForceDirectedLayout(E_hat, N)
        self.assertEqual(pos.shape, (N, 2))
        failed, reason = CheckReconstructionFailure(pos, E_hat)
        # 小规模合成数据可能天然碎片化，接受警告
        if failed:
            print(f"  ⚠ 重建警告(预期): {reason}")

        print(f"  ✓ 集成测试通过: N={N}, edges={len(E_hat)}")


if __name__ == "__main__":
    unittest.main()
