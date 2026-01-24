# tools/plot_ablation_metrics.py
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
OUTPUT_DIR = "results/paper_figures"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置学术风格 (Whitegrid + Paper Context)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_multi_metric_ablation():
    """
    绘制 5 个模型配置在 4 个指标上的分组柱状图
    """
    # 1. 准备数据
    # 注意：列顺序对应 [Proposed, No-CBAM, Uni-GRU, Z-Order, Dot-Attn]
    raw_data = [
        # Proposed (base_row_add)
        {'Config': 'Proposed', 'Metric': 'Accuracy', 'Value': 0.9986},
        {'Config': 'Proposed', 'Metric': 'Precision', 'Value': 0.9988},
        {'Config': 'Proposed', 'Metric': 'Recall', 'Value': 0.9984},
        {'Config': 'Proposed', 'Metric': 'F1-Score', 'Value': 0.9986},

        # No-CBAM
        {'Config': 'No-CBAM', 'Metric': 'Accuracy', 'Value': 0.9882},
        {'Config': 'No-CBAM', 'Metric': 'Precision', 'Value': 0.9908},
        {'Config': 'No-CBAM', 'Metric': 'Recall', 'Value': 0.9869},
        {'Config': 'No-CBAM', 'Metric': 'F1-Score', 'Value': 0.9888},

        # Uni-GRU
        {'Config': 'Uni-GRU', 'Metric': 'Accuracy', 'Value': 0.9728},
        {'Config': 'Uni-GRU', 'Metric': 'Precision', 'Value': 0.9719},
        {'Config': 'Uni-GRU', 'Metric': 'Recall', 'Value': 0.9766},
        {'Config': 'Uni-GRU', 'Metric': 'F1-Score', 'Value': 0.9742},

        # Z-Order (var_z_add)
        {'Config': 'Z-Order', 'Metric': 'Accuracy', 'Value': 0.9973},
        {'Config': 'Z-Order', 'Metric': 'Precision', 'Value': 0.9971},
        {'Config': 'Z-Order', 'Metric': 'Recall', 'Value': 0.9978},
        {'Config': 'Z-Order', 'Metric': 'F1-Score', 'Value': 0.9975},

        # Dot-Attn (var_row_dot)
        {'Config': 'Dot-Attn', 'Metric': 'Accuracy', 'Value': 0.9947},
        {'Config': 'Dot-Attn', 'Metric': 'Precision', 'Value': 0.9956},
        {'Config': 'Dot-Attn', 'Metric': 'Recall', 'Value': 0.9944},
        {'Config': 'Dot-Attn', 'Metric': 'F1-Score', 'Value': 0.9950},
    ]

    df = pd.DataFrame(raw_data)

    # 2. 设定绘图顺序
    config_order = ['Proposed', 'No-CBAM', 'Uni-GRU', 'Z-Order', 'Dot-Attn']
    metric_order = ['F1-Score', 'Accuracy', 'Precision', 'Recall']  # 调整图例顺序

    # 3. 创建画布
    plt.figure(figsize=(18, 9))  # 稍微宽一点，容纳分组柱子

    # 4. 定义四种指标的配色 (学术风：深红、深蓝、橙色、青色)
    palette = {
        'F1-Score': '#d62728',  # Red
        'Accuracy': '#1f77b4',  # Blue
        'Precision': '#ff7f0e',  # Orange
        'Recall': '#2ca02c',  # Green
    }

    # 5. 绘制分组柱状图
    ax = sns.barplot(
        x='Config',
        y='Value',
        hue='Metric',
        data=df,
        order=config_order,
        hue_order=metric_order,
        palette=palette,
        edgecolor='black',
        linewidth=1.0,
        width=0.8  # 调整柱子组的宽度
    )

    # 6. 关键：截断 Y 轴以放大差异
    # [修改] 上限调至 1.005，防止 0.9989 这种数值的标签被顶部截断
    plt.ylim(0.97, 1.000)

    # 7. 图表装饰
    plt.title('Performance Comparison of Different Ablation Configurations', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('')  # Config名字已经在X轴刻度上了，不需要额外标签

    # 8. 图例设置 (移动到右侧外部或上方，避免遮挡数值)
    plt.legend(
        title='Metrics',
        title_fontsize=14,
        fontsize=13,
        loc='upper left',
        bbox_to_anchor=(1.0, 1.0),  # 放在图表右侧外
        frameon=True,
        shadow=True,
        ncol=1
    )

    # 9. [修改] 给所有柱子标上数值
    for container in ax.containers:
        # 移除之前的 if 判断，直接对所有 container 进行标注
        # 使用 rotation=90 垂直显示数值，防止重叠
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

    # 10. 保存
    save_path = os.path.join(OUTPUT_DIR, "fig_ablation_multi_metric.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 多指标消融实验对比图已保存: {save_path}")


if __name__ == "__main__":
    print("🚀 开始绘制升级版图表...")
    plot_multi_metric_ablation()
    print("\n🎉 绘图完成！请查看 results/paper_figures 文件夹。")