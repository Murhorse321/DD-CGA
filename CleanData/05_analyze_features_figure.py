# 05_analyze_features_vis.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ================= 🧪 实验配置 =================
TRAIN_FILE = "data/step4_split/train.csv"
OUTPUT_DIR = "results/feature_analysis"  # 图表保存路径
REPORT_FILE = os.path.join(OUTPUT_DIR, "feature_report.csv")

# 绘图配置
TOP_N_PLOT = 20  # 在柱状图中展示前多少个特征
TARGET_FEAT_NUM = 64  # 我们的目标特征数 (用于在图中画截断线)
DPI = 300  # 图片分辨率 (300为学术打印标准)


# 为了论文通用性，图表标签建议使用英文
# 如果需要中文，请解开下面两行的注释，并确保系统有 SimHei 字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_importance_bar(df_report, output_dir):
    """绘制 Top N 特征重要性柱状图"""
    plt.figure(figsize=(10, 8))

    # 取前 N 个
    top_data = df_report.head(TOP_N_PLOT).sort_values(by='Importance', ascending=True)

    # 绘制水平柱状图
    bars = plt.barh(top_data['Feature'], top_data['Importance'], color='#3498db', edgecolor='black', alpha=0.7)

    plt.xlabel('Gini Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {TOP_N_PLOT} Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # 保存
    save_path = os.path.join(output_dir, "feature_importance_top20.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"   📊 [图表1] 特征重要性柱状图已保存: {save_path}")


def plot_cumulative_curve(df_report, output_dir):
    """绘制累积重要性曲线，证明前64个特征足够重要"""
    plt.figure(figsize=(10, 6))

    # 计算累积和
    cumulative_importances = np.cumsum(df_report['Importance'])
    x_values = np.arange(len(cumulative_importances)) + 1

    plt.plot(x_values, cumulative_importances, 'r-', linewidth=2, label='Cumulative Importance')
    plt.fill_between(x_values, cumulative_importances, color='red', alpha=0.1)

    # 标记我们的截断点 (64)
    if len(df_report) >= TARGET_FEAT_NUM:
        cum_score_64 = cumulative_importances[TARGET_FEAT_NUM - 1]
        plt.axvline(x=TARGET_FEAT_NUM, color='blue', linestyle='--', label=f'Cut-off @ {TARGET_FEAT_NUM} Features')
        plt.axhline(y=cum_score_64, color='blue', linestyle='--', alpha=0.5)
        plt.text(TARGET_FEAT_NUM + 2, cum_score_64 - 0.05, f'{cum_score_64:.1%} Explained', color='blue',
                 fontweight='bold')

    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Cumulative Importance', fontsize=12)
    plt.title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(output_dir, "feature_cumulative_importance.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"   📊 [图表2] 累积重要性曲线已保存: {save_path}")


def plot_correlation_heatmap(df, top_features, output_dir):
    """仅绘制 Top 15 特征的相关性热力图 (全量画太乱)"""
    plt.figure(figsize=(12, 10))

    # 提取 Top 15 特征的数据
    top_15 = top_features[:15]
    corr_matrix = df[top_15].corr()

    # 绘制热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只画下三角
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlation Matrix of Top 15 Features', fontsize=14, fontweight='bold')

    save_path = os.path.join(output_dir, "feature_correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"   📊 [图表3] Top特征相关性热力图已保存: {save_path}")


def main():
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ 错误: 找不到文件 {TRAIN_FILE}")
        return

    ensure_dir(OUTPUT_DIR)

    print("🚀 开始特征分析与可视化...")
    df = pd.read_csv(TRAIN_FILE)

    # 准备数据 (排除标签)
    exclude_cols = ['label', 'label_int']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['label_int']

    features = X.columns.tolist()
    print(f"   分析特征数: {len(features)}")

    # 1. 训练随机森林
    print("   正在计算特征重要性...")
    rf = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X, y)

    # 2. 生成报告
    importances = rf.feature_importances_
    report = pd.DataFrame({'Feature': features, 'Importance': importances})
    report = report.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    report['Rank'] = report.index + 1

    # 保存 CSV
    report.to_csv(REPORT_FILE, index=False)
    print(f"   💾 CSV 报告已保存: {REPORT_FILE}")

    # 3. 生成图表
    print("   正在生成学术级图表...")

    # 图1: 柱状图
    plot_importance_bar(report, OUTPUT_DIR)

    # 图2: 累积曲线
    plot_cumulative_curve(report, OUTPUT_DIR)

    # 图3: 热力图 (需要传入原始数据 X 用于计算相关性)
    plot_correlation_heatmap(X, report['Feature'].tolist(), OUTPUT_DIR)

    print("-" * 50)
    print("✅ 分析完成！请查看 results/feature_analysis 文件夹。")
    print("   请将生成的 'feature_importance_top20.png' 发给我，或者复制 CSV 中的前20行。")
    print("   我们即将决定最终的 64 个特征。")


if __name__ == "__main__":
    main()