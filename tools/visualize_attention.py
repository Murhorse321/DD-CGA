# # tools/visualize_attention_final.py
# # -*- coding: utf-8 -*-
# import os
# import argparse
# import yaml
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 尝试导入模型
# try:
#     from models.cnn_gru_attn import CNNGRUAttn
# except ImportError:
#     import sys
#
#     sys.path.append(".")
#     from models.cnn_gru_attn import CNNGRUAttn
#
#
# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def main(args):
#     device = get_device()
#
#     # 1. 加载配置
#     print(f"📖 Loading config: {args.config}")
#     with open(args.config, 'r', encoding='utf-8') as f:
#         cfg = yaml.safe_load(f)
#
#     # 2. 初始化模型
#     print(f"▶ Loading Model...")
#     mcfg = cfg['model']
#     model = CNNGRUAttn(
#         num_classes=int(mcfg.get('num_classes', 1)),
#         cnn_channels=tuple(mcfg.get('cnn_channels', [32, 64])),
#         use_cbam=bool(mcfg.get('use_cbam', True)),
#         cbam_reduction=int(mcfg.get('cbam_reduction', 8)),
#         gru_hidden=int(mcfg.get('gru_hidden', 128)),
#         gru_layers=int(mcfg.get('gru_layers', 1)),
#         bidirectional=bool(mcfg.get('bidirectional', False)),
#         attn_type=str(mcfg.get('attn_type', 'add')),
#         dropout=float(mcfg.get('dropout', 0.5)),
#         use_batchnorm=bool(mcfg.get('use_batchnorm', True)),
#         sequence_order=str(mcfg.get('sequence_order', 'row')),
#         temperature=float(mcfg.get('temperature', 1.0)),
#     ).to(device)
#
#     # 3. 加载权重
#     print(f"▶ Loading Weights: {args.ckpt}")
#     checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
#     state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#
#     # 4. 准备数据
#     test_path = cfg['data']['test_path']
#     print(f"📖 Reading Test Data: {test_path}")
#     df = pd.read_csv(test_path)
#
#     # 输出目录
#     fig_save_dir = "results/paper_figures/heatmaps"
#     os.makedirs(fig_save_dir, exist_ok=True)
#
#     # 设定学术绘图风格
#     sns.set_theme(style="white", context="paper", font_scale=1.5)
#
#     # 定义目标
#     targets = ['Benign', 'Portmap', 'Syn']
#
#     print(f"🎨 Generating Heatmaps for: {targets}")
#
#     for attack_name in targets:
#         # --- 智能筛选样本 ---
#         if attack_name == 'Benign':
#             # 找 label_int=0 且预测正确的样本（True Negative）
#             # 简单起见，我们取 label_int=0 的第一个
#             samples = df[df['label_int'] == 0].head(1)
#         else:
#             # 找对应攻击名字的样本
#             if 'label' in df.columns:
#                 samples = df[df['label'] == attack_name].head(1)
#             else:
#                 print(f"⚠️ CSV 缺少 'label' 列，无法按名字筛选 {attack_name}")
#                 continue
#
#         if samples.empty:
#             print(f"⚠️ No samples found for {attack_name}")
#             continue
#
#         # --- 预处理 ---
#         ignore_cols = ['label', 'label_int']
#         feature_cols = [c for c in df.columns if c not in ignore_cols]
#         X_numpy = samples[feature_cols].values.astype(np.float32)[:, :64]  # 确保64维
#         X_tensor = torch.from_numpy(X_numpy).reshape(1, 1, 8, 8).to(device)
#
#         # --- 推理 ---
#         with torch.no_grad():
#             logits, attn_weights = model(X_tensor, return_attn=True)
#             pred_prob = torch.sigmoid(logits).item()
#             weights_np = attn_weights.cpu().numpy().squeeze()
#
#         # --- 绘图 ---
#         heatmap_data = weights_np.reshape(4, 4)
#
#         plt.figure(figsize=(6, 5))
#
#         # 【关键修改】：使用 coolwarm 配色，且固定 vmax=1.0
#         # 这样 Benign (权重低) 会偏蓝/白，Attack (权重高) 会偏红
#         ax = sns.heatmap(
#             heatmap_data,
#             annot=True,
#             fmt=".2f",
#             cmap="coolwarm",  # 冷暖色调
#             vmin=0,
#             vmax=1.0,  # 强制最大值为 1.0 (Attention 上限)
#             cbar=True,
#             square=True,
#             linewidths=1,
#             linecolor='black'
#         )
#
#         plt.title(f"Class: {attack_name}\nModel Prob: {pred_prob:.4f}", fontsize=14, fontweight='bold')
#         plt.axis('off')  # 去掉坐标轴刻度，更像图片
#
#         save_path = os.path.join(fig_save_dir, f"heatmap_{attack_name}.png")
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         plt.close()
#         print(f"  ✅ Saved: {save_path}")
#
#     print(f"\n🎉 所有热力图已生成: {fig_save_dir}")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config/cnn_gru_att.yaml")
#     # 请手动填入你的最佳权重路径
#     parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
#     args = parser.parse_args()
#     main(args)

# tools/visualize_attention_final.py
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 尝试导入模型
try:
    from models.cnn_gru_attn import CNNGRUAttn
except ImportError:
    sys.path.append(".")
    from models.cnn_gru_attn import CNNGRUAttn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    device = get_device()

    # =========================
    # 1. 加载配置
    # =========================
    print(f"📖 Loading config: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # =========================
    # 2. 初始化模型
    # =========================
    print("▶ Loading Model...")
    mcfg = cfg['model']
    model = CNNGRUAttn(
        num_classes=int(mcfg.get('num_classes', 1)),
        cnn_channels=tuple(mcfg.get('cnn_channels', [32, 64])),
        use_cbam=bool(mcfg.get('use_cbam', True)),
        cbam_reduction=int(mcfg.get('cbam_reduction', 8)),
        gru_hidden=int(mcfg.get('gru_hidden', 128)),
        gru_layers=int(mcfg.get('gru_layers', 1)),
        bidirectional=bool(mcfg.get('bidirectional', False)),
        attn_type=str(mcfg.get('attn_type', 'add')),
        dropout=float(mcfg.get('dropout', 0.5)),
        use_batchnorm=bool(mcfg.get('use_batchnorm', True)),
        sequence_order=str(mcfg.get('sequence_order', 'row')),
        temperature=float(mcfg.get('temperature', 1.0)),
    ).to(device)

    # =========================
    # 3. 加载权重
    # =========================
    print(f"▶ Loading Weights: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # =========================
    # 4. 读取测试数据
    # =========================
    test_path = cfg['data']['test_path']
    print(f"📖 Reading Test Data: {test_path}")
    # 建议只读取需要的列以加快速度，或者全读
    df = pd.read_csv(test_path)

    # 输出目录
    fig_save_dir = "results/paper_figures/attention_heatmaps"
    os.makedirs(fig_save_dir, exist_ok=True)

    # =========================
    # 5. 学术绘图风格设置
    # =========================
    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.4
    )

    # =========================
    # 6. 需要可视化的类别
    # =========================
    targets = ['Benign', 'Portmap', 'Syn']
    print(f"🎨 Generating attention heatmaps for: {targets}")

    for attack_name in targets:

        # =========================
        # 7. 样本选择
        # =========================
        if attack_name == 'Benign':
            # 找 label_int 为 0 的样本
            samples = df[df['label_int'] == 0].sample(n=1, random_state=42)
        else:
            if 'label' not in df.columns:
                print(f"⚠️ Missing 'label' column, skip {attack_name}")
                continue
            # 找对应攻击名称的样本
            samples = df[df['label'] == attack_name].sample(n=1, random_state=42)

        if samples.empty:
            print(f"⚠️ No samples found for {attack_name}")
            continue

        # =========================
        # 8. 特征预处理
        # =========================
        ignore_cols = ['label', 'label_int']
        feature_cols = [c for c in df.columns if c not in ignore_cols]

        # 假设前 64 个特征
        X_numpy = samples[feature_cols].values.astype(np.float32)[:, :64]
        # Reshape 为 (Batch, Channel, H, W) -> (1, 1, 8, 8)
        X_tensor = torch.from_numpy(X_numpy).reshape(1, 1, 8, 8).to(device)

        # =========================
        # 9. 推理与 Attention 获取
        # =========================
        with torch.no_grad():
            logits, attn_weights = model(X_tensor, return_attn=True)
            pred_prob = torch.sigmoid(logits).item()
            weights_np = attn_weights.detach().cpu().numpy().squeeze()

        # =========================
        # 10. Attention 归一化
        # =========================
        # 如果需要归一化到 0-1 之间以便绘图对比
        if weights_np.max() > 1.0 or weights_np.min() < 0.0:
            weights_np = (weights_np - weights_np.min()) / (
                    weights_np.max() - weights_np.min() + 1e-8
            )

        # 假设 Attention 输出是 16 (4x4)
        heatmap_data = weights_np.reshape(4, 4)

        # =========================
        # 11. 绘制 Attention Heatmap
        # =========================
        plt.figure(figsize=(5.5, 5))

        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="viridis",  # 你选择的配色
            vmin=0,
            vmax=1.0,  # 固定最大值为 1，便于横向对比
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="gray"
        )

        plt.title(f"Class: {attack_name}\nProb: {pred_prob:.4f}", fontsize=14, fontweight='bold')
        plt.axis('off')

        # 保存
        save_path = os.path.join(fig_save_dir, f"heatmap_{attack_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  ✅ Saved: {save_path}")

    print(f"\n🎉 All Done! Figures saved in: {fig_save_dir}")


# =========================
# ★★★ 修复的核心：程序入口 ★★★
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/cnn_gru_att.yaml", help="Path to config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt checkpoint")
    args = parser.parse_args()

    main(args)