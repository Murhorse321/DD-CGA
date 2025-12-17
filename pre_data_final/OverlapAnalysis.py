import pandas as pd
import os

# 📂 数据集路径（请修改为你的实际路径）
csv_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\final_balanced_dataset_onehot_Pro.csv'

# 读取数据集
print(f"📥 正在读取数据集：{csv_path}")
df = pd.read_csv(csv_path)

# 分别获取正常流量和攻击流量
normal_df = df[df['label'] == 0]
attack_df = df[df['label'] == 1]

# 获取所有特征列（不包含标签列）
feature_cols = [col for col in df.columns if col != 'label']

print(f"📊 开始重叠度分析，特征总数: {len(feature_cols)}")

overlap_results = []

for feature in feature_cols:
    normal_values = set(normal_df[feature].unique())
    attack_values = set(attack_df[feature].unique())

    intersection = normal_values & attack_values
    union = normal_values | attack_values

    if len(union) == 0:
        overlap = 0.0
    else:
        overlap = len(intersection) / len(union)

    overlap_results.append((feature, overlap))

# 排序，重叠度从低到高
overlap_results.sort(key=lambda x: x[1])

# 输出重叠度结果
print("\n📋 特征重叠度分析结果：")
for feature, overlap in overlap_results:
    print(f"{feature:30} | 重叠度: {overlap:.4f}")

# 保存结果
result_df = pd.DataFrame(overlap_results, columns=['Feature', 'Overlap'])
result_df.to_csv('feature_overlap_analysis.csv', index=False)
print("\n✅ 重叠度分析结果已保存到 feature_overlap_analysis.csv")
