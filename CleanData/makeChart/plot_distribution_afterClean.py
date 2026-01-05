
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
# 我们读取 Step 5 或 Step 6 的数据均可，因为样本数没变
# 这里读取 Step 5 (Final Features)，因为那是刚完成筛选的状态
INPUT_DIR = "D:\Desktop\C_G_A\CNN_GRU_ATTENTION\CleanData\data\step5_final"
OUTPUT_FILE = "dataset_distribution.png"

# 绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 尝试支持中文，如果乱码请改回 Arial
plt.rcParams['axes.unicode_minus'] = False


# =======================================

def main():
    if not os.path.exists(INPUT_DIR):
        print("❌ 找不到数据目录，请确认 Step 3.5 已完成。")
        return

    # 1. 读取 Train, Val, Test 并合并统计
    # 我们要展示的是【整个实验数据集】的构成
    dfs = []
    for t in ['train.csv', 'val.csv', 'test.csv']:
        path = os.path.join(INPUT_DIR, t)
        if os.path.exists(path):
            print(f"📖 读取 {t} ...")
            dfs.append(pd.read_csv(path))

    if not dfs:
        return

    full_df = pd.concat(dfs, ignore_index=True)
    total_samples = len(full_df)
    print(f"📊 数据集总样本量: {total_samples}")

    # 2. 统计各类别数量
    # label 列是字符串名称 (例如 'DrDoS_DNS', 'Benign')
    counts = full_df['label'].value_counts()

    # 3. 绘图
    plt.figure(figsize=(12, 8), dpi=300)

    # 定义颜色：良性用绿色，攻击用红色系
    # 先获取所有类别名
    labels = counts.index.tolist()
    colors = ['#2ecc71' if 'Benign' in lbl else '#e74c3c' for lbl in labels]

    # 画柱状图
    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors)

    # 设置标签
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f'Distribution of Traffic Categories in Constructed Dataset (Total: {total_samples})', fontsize=14,
              fontweight='bold')
    plt.xlabel('Traffic Category', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)

    # 在柱子上标数值
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 100,
                f'{int(height)}',
                ha="center", va="bottom", fontsize=9)

    # 保存
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"✅ 图表已保存至: {OUTPUT_FILE}")
    print("   请将此图插入论文 3.2 节，与原始分布图形成对比。")


if __name__ == "__main__":
    main()