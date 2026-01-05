# 11_plot_reshape_diagram_v2.py
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as patches

# ================= 配置 =================
OUTPUT_FILE = "reshape_diagram_with_arrow.png"


# =======================================

def main():
    # 1. 准备数据
    data_1d = np.linspace(0, 1, 64).reshape(1, 64)
    data_2d = data_1d.reshape(8, 8)

    # 2. 创建画布：调整为更宽的比例，方便放箭头
    # 使用 GridSpec 将画布分为三部分：左图(5份) - 箭头(2份) - 右图(5份)
    fig = plt.figure(figsize=(14, 5), dpi=300)
    gs = fig.add_gridspec(1, 11)  # 将画布横向切成12份

    # --- 左图：1D 向量 ---
    ax1 = fig.add_subplot(gs[0, 0:5])  # 占前5份
    sns.heatmap(data_1d, ax=ax1, cmap="Blues", cbar=False,
                xticklabels=False, yticklabels=False, square=False, linewidths=0.5, linecolor='white')
    ax1.set_title("Input: 1D Feature Vector\n(1 × 64)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Feature Index (1 -> 64)", fontsize=11)
    # 标注首尾
    ax1.text(0.5, 0.5, 'x_1', ha='center', va='center', fontsize=9, color='black', weight='bold')
    ax1.text(64, 0.5, 'x_64', ha='center', va='center', fontsize=9, color='black', weight='bold')

    # --- 中间：大箭头 ---
    ax_arrow = fig.add_subplot(gs[0, 5:7])  # 占中间2份
    ax_arrow.axis('off')  # 关掉坐标轴

    # 绘制箭头 (FancyArrow)
    # (x, y) 是起点，(dx, dy) 是长度和方向
    arrow = patches.FancyArrow(0.1, 0.5, 0.8, 0, width=0.15, head_width=0.4, head_length=0.3,
                               length_includes_head=True, color='#34495e')  # 深灰色箭头
    ax_arrow.add_patch(arrow)

    # 在箭头上方加文字
    ax_arrow.text(0.5, 0.75, "Reshape", ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax_arrow.text(0.5, 0.25, "Mapping", ha='center', va='center', fontsize=10, style='italic', color='gray')

    # --- 右图：2D 矩阵 ---
    ax2 = fig.add_subplot(gs[0, 7:11])  # 占后5份
    sns.heatmap(data_2d, ax=ax2, cmap="Blues", cbar=True,
                xticklabels=False, yticklabels=False, square=True, linewidths=0.5, linecolor='white',
                cbar_kws={"shrink": 0.8, "label": "Normalized Value"})
    ax2.set_title("Output: 2D Image Matrix\n(8 × 8)", fontsize=14, fontweight='bold', pad=15)

    # 标注关键点，体现重塑顺序
    # 字体颜色根据背景深浅调整
    ax2.text(0.5, 0.5, 'x_1', ha='center', va='center', fontsize=10, color='black', weight='bold')
    ax2.text(7.5, 0.5, 'x_8', ha='center', va='center', fontsize=10, color='black', weight='bold')
    ax2.text(0.5, 1.5, 'x_9', ha='center', va='center', fontsize=10, color='black', weight='bold')
    ax2.text(7.5, 7.5, 'x_64', ha='center', va='center', fontsize=10, color='black', weight='bold')

    # 3. 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"✅ 带箭头示意图已生成: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()