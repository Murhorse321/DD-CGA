# tools/visualize_attention.py
# -*- coding: utf-8 -*-
# python tools/visualize_attention.py \
#   --config config/cnn_gru_att.yaml \
#   --ckpt results/tuning_gru_attn/ATT_20260106-180052/ckpt/checkpoint_best.pt
# 可视化观察模型是否聚焦到不同类别的关键信息（热力图）
# tools/visualize_attention.py
# -*- coding: utf-8 -*-
import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入模型
try:
    from models.cnn_gru_attn import CNNGRUAttn
except ImportError:
    import sys

    sys.path.append(".")
    from models.cnn_gru_attn import CNNGRUAttn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    device = get_device()

    # 1. 加载配置
    print(f"📖 Loading config: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 初始化模型
    print(f"▶ Loading Model structure...")
    mcfg = cfg['model']
    # 确保参数与训练时一致
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

    # 3. 加载权重
    print(f"▶ Loading Weights from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)

    # 兼容处理：检查 checkpoint 是不是包含 'model' 键
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"⚠️ Weight loading mismatch (trying non-strict mode): {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("✅ Model loaded successfully.")

    # 4. 准备数据
    test_path = cfg['data']['test_path']
    print(f"📖 Reading test data: {test_path}")
    # 读取整个 CSV（如果很大，可以优化为只读部分，但为了找特定类别先全读）
    df = pd.read_csv(test_path)

    # 定义目标攻击类型
    target_attacks = ['Portmap', 'Syn', 'Benign', 'DrDoS_DNS']

    fig_save_dir = os.path.join(os.path.dirname(args.ckpt), "..", "vis_heatmaps")
    os.makedirs(fig_save_dir, exist_ok=True)

    print(f"🎨 Generating Heatmaps for: {target_attacks}")

    # tools/visualize_attention.py (修正片段)

    for attack_name in target_attacks:
        # --- 样本筛选 (修正版) ---
        if attack_name == 'Benign':
            # 优先尝试通过 label_int = 0 来找
            if 'label_int' in df.columns:
                samples = df[df['label_int'] == 0].head(1)
            # 如果没有 label_int，尝试匹配字符串 "Benign"
            elif 'label' in df.columns:
                samples = df[df['label'] == 'Benign'].head(1)
            else:
                print("⚠️ Cannot find Benign samples (no label_int=0 or label='Benign')")
                continue
        else:
            # 对于攻击类型，通过 label 列的字符串匹配
            if 'label' in df.columns:
                samples = df[df['label'] == attack_name].head(1)
            else:
                print(f"⚠️ Cannot find string label column for {attack_name}")
                continue

        if samples.empty:
            print(f"⚠️ No samples found for {attack_name} (Check column names/values)")
            continue

        # ... (后续预处理和绘图代码保持不变) ...

        # --- 数据预处理 ---
        # 排除非特征列
        ignore_cols = ['label', 'label_int']
        feature_cols = [c for c in df.columns if c not in ignore_cols]

        # 提取数值并转 Tensor
        X_numpy = samples[feature_cols].values.astype(np.float32)
        # 确保只取前 64 维 (8x8)
        if X_numpy.shape[1] > 64:
            X_numpy = X_numpy[:, :64]

        X_tensor = torch.from_numpy(X_numpy).reshape(1, 1, 8, 8).to(device)

        # --- 核心：利用模型自带的 return_attn ---
        with torch.no_grad():
            # 这里调用 forward(x, return_attn=True)
            logits, attn_weights = model(X_tensor, return_attn=True)

            # 预测概率
            pred_prob = torch.sigmoid(logits).item()

            # attn_weights shape: [B, 16] -> [1, 16]
            weights_np = attn_weights.cpu().numpy().squeeze()

        # --- 绘图 ---
        # Reshape to 4x4
        heatmap_data = weights_np.reshape(4, 4)

        plt.figure(figsize=(6, 5))
        # 使用 Reds 色系，vmin=0 确保底色一致
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=heatmap_data.max())

        plt.title(f"Type: {attack_name}\nPred: {pred_prob:.4f}")

        # plt.figure(figsize=(6, 5))
        #
        # # [修改点 1] 固定 vmax = 1.0 (或 0.5)，不再跟随样本变化。
        # # 这样 0.07 的权重就会显示为非常淡的粉色/接近白色，而 0.8 才会显示为深红。
        # # [修改点 2] 如果你想要“蓝色”表示低值，“红色”表示高值，可以用 cmap="coolwarm"
        #
        # sns.heatmap(heatmap_data,
        #             annot=True,
        #             fmt=".2f",
        #             cmap="coolwarm",  # 改为冷暖色调：蓝色低，红色高
        #             vmin=0,
        #             vmax=1.0)  # 固定最大值为 1.0 (Attention 上限)
        #
        # plt.title(f"Type: {attack_name}\nPred: {pred_prob:.4f}")

        save_path = os.path.join(fig_save_dir, f"{attack_name}_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  -> Saved: {save_path}")




    print(f"\n✅ All Done! Images saved in: {fig_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/cnn_gru_att.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best checkpoint")
    args = parser.parse_args()
    main(args)