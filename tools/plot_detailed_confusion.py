# # tools/plot_detailed_confusion.py
# # -*- coding: utf-8 -*-
#
# import pandas as pd
# import argparse
# import os
# import yaml
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # 设置学术绘图风格
# sns.set_theme(style="white", context="paper", font_scale=1.2)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Plot N x 2 Detailed Confusion Matrix")
#     parser.add_argument("--pred_file", type=str, required=True,
#                         help="训练生成的 test_preds.csv 路径")
#     parser.add_argument("--config", type=str, default="config/cnn_gru_att.yaml",
#                         help="用于查找原始测试集路径的配置文件")
#     parser.add_argument("--output_dir", type=str, default="results/paper_figures",
#                         help="图片保存目录")
#     args = parser.parse_args()
#
#     # 1. 确定原始测试集路径
#     if os.path.exists(args.config):
#         with open(args.config, 'r', encoding='utf-8') as f:
#             cfg = yaml.safe_load(f)
#         test_path = cfg.get('data', {}).get('test_path')
#     else:
#         print(f"❌ Config not found: {args.config}")
#         return
#
#     if not test_path or not os.path.exists(test_path):
#         print(f"❌ Test data not found: {test_path}")
#         return
#
#     # 2. 读取数据
#     print(f"📖 Reading Predictions: {args.pred_file}")
#     df_pred = pd.read_csv(args.pred_file)  # columns: y_true, y_prob, y_pred
#
#     print(f"📖 Reading Original Labels: {test_path}")
#     # 我们需要原始的 label (字符串)
#     df_orig = pd.read_csv(test_path, usecols=['label'])
#
#     # 对齐数据（处理可能的长度不一致，通常是因为 drop_last）
#     min_len = min(len(df_pred), len(df_orig))
#     df_pred = df_pred.iloc[:min_len]
#     df_orig = df_orig.iloc[:min_len]
#
#     # 合并
#     df = pd.concat([df_orig.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
#
#     # 3. 构造 N x 2 混淆矩阵数据
#     # 我们统计每种 label 被预测为 0 (Benign) 和 1 (Attack) 的数量
#     print("📊 Aggregating data...")
#     pivot_data = df.groupby('label')['y_pred'].value_counts().unstack(fill_value=0)
#
#     # 确保列名为 [0, 1] (即 Pred: Benign, Pred: Attack)
#     if 0 not in pivot_data.columns: pivot_data[0] = 0
#     if 1 not in pivot_data.columns: pivot_data[1] = 0
#     pivot_data = pivot_data[[0, 1]]  # 调整列顺序
#
#     # 排序：
#     # 1. 把 Benign 放在第一行
#     # 2. 其他攻击类型按 "漏报率" (预测为0的比例) 排序，漏报越多的越靠前，方便审稿人看问题
#     pivot_data['Error_Rate'] = pivot_data[0] / (pivot_data[0] + pivot_data[1])
#
#     # 分离 Benign 和 Attack
#     if 'Benign' in pivot_data.index:
#         benign_row = pivot_data.loc[['Benign']]
#         attack_rows = pivot_data.drop('Benign').sort_values(by='Error_Rate', ascending=False)
#         final_df = pd.concat([benign_row, attack_rows])
#     else:
#         final_df = pivot_data.sort_values(by='Error_Rate', ascending=False)
#
#     # 移除辅助列
#     plot_data = final_df[[0, 1]]
#
#     # 4. 绘图
#     plt.figure(figsize=(10, len(plot_data) * 0.5 + 2))  # 根据行数动态调整高度
#
#     # 使用 Log Norm 颜色映射，因为大类(DrDoS_NTP)可能有数万条，小类只有几百条
#     # Log Norm 能让小数值（误判的几个样本）也能有颜色显示
#     from matplotlib.colors import LogNorm
#
#     # 为了防止 log(0) 报错，加一个微小值或者使用 linear (如果数据量级差异没那么大)
#     # 这里我们用线性颜色，但在文字标注上做文章
#
#     ax = sns.heatmap(
#         plot_data,
#         annot=True,
#         fmt="d",  # 显示整数数量
#         cmap="Reds",  # 红色系：颜色越深代表数量越多
#         cbar=True,
#         linewidths=0.5,
#         linecolor='black'
#     )
#
#     # 5. 调整标签
#     plt.title('Detailed Confusion Matrix: True Labels vs. Prediction', fontsize=14, fontweight='bold', pad=20)
#     plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
#     plt.ylabel('True Traffic Category', fontsize=12, fontweight='bold')
#
#     # 修改 X 轴刻度标签
#     ax.set_xticklabels(['Benign (Safe)', 'Attack (DDoS)'], fontsize=11)
#
#     # 6. 特别标注：把“错误”的格子圈出来？
#     # 对于 Benign 行，[0]是常态，[1]是误报(FP) -> 重点关注 [1]
#     # 对于 Attack 行，[1]是常态，[0]是漏报(FN) -> 重点关注 [0]
#     # 这里我们依靠读者的直觉：对角线通常是对的，非对角线是错的。
#     # 但由于这是 Nx2，逻辑略有不同：
#     # 第一行(Benign): 左边对，右边错。
#     # 其他行(Attack): 右边对，左边错。
#
#     # 保存
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     save_path = os.path.join(args.output_dir, "fig_detailed_confusion_matrix.png")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     print(f"\n✅ 详细混淆矩阵已保存: {save_path}")
#     print("💡 解读指南:")
#     print("   - 第一行 (Benign): 右侧格子数值代表误报数 (False Positives)。")
#     print("   - 其他行 (Attack): 左侧格子数值代表漏报数 (False Negatives)。")
#     print("   - 重点检查 Portmap 左侧格子的数字。")
#
#
# if __name__ == "__main__":
#     main()
#
#
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. 类别名称（保持不变）
# =========================
true_labels = [
    "Benign",
    "Portmap",
    "NetBIOS",
    "DrDoS_SNMP",
    "LDAP",
    "DrDoS_NTP",
    "DrDoS_DNS",
    "Syn",
    "UDPLag",
    "MSSQL",
    "DrDoS_SSDP",
    "TFTP",
    "UDP"
]

predicted_labels = ["Benign (Safe)", "Attack (DDoS)"]

# =========================
# 2. 混淆矩阵数据（你只需要改这里）
# 行：真实类别
# 列：预测类别
# =========================
confusion_matrix = np.array([
    [10475, 18],
    [2,     210],
    [3,     1005],
    [2,     981],
    [1,     1007],
    [1,     1029],
    [2,     971],
    [1,     981],
    [0,     1017],
    [1,     1052],
    [0,     1002],
    [0,     966],
    [0,     988]
])

# =========================
# 3. 画图
# =========================
plt.figure(figsize=(10, 8))

sns.heatmap(
    confusion_matrix,
    annot=True,           # 显示数值
    fmt="d",               # 整数格式
    cmap="Reds",           # 颜色风格
    cbar=True,             # 显示颜色条
    xticklabels=predicted_labels,
    yticklabels=true_labels
)

# =========================
# 4. 图形细节设置
# =========================
plt.title("Confusion Matrix of True and Predicted Labels", fontsize=14)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Traffic Category", fontsize=12)

plt.tight_layout()
plt.show()
