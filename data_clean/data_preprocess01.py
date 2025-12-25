import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 📍 数据路径
file_path = 'E:/CIC-DDoS/CSVs/CICDDoS2019_Merged.csv'

# 📁 输出目录
output_dir = 'processed_chunks'
os.makedirs(output_dir, exist_ok=True)

# ⚙️ 分块大小
chunk_size = 50000

# 🚫 无用列列表（根据需要可继续扩展）
useless_columns = [
    'flow id', 'source ip', 'source port', 'destination ip', 'destination port',
    'timestamp', 'simillarhttp', 'inbound', 'unnamed: 0',
     # ✅ 新增删除列
]

# ✅ 初始化工具
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

print("📥 正在读取并分块处理数据...")

for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, encoding='utf-8')):
    print(f"\n📦 正在处理第 {i+1} 块数据...")

    # ✂️ 清洗列名
    chunk.columns = chunk.columns.str.strip().str.lower()

    # 🕵️‍♂️ 检查标签列是否存在
    if 'label' not in chunk.columns:
        print(f"⚠️ 第 {i+1} 块跳过，未找到 label 列。")
        continue

    # 🚮 删除无用列
    chunk = chunk.drop(columns=[col for col in useless_columns if col in chunk.columns], errors='ignore')

    # 🧹 删除全为空的列
    chunk = chunk.dropna(axis=1, how='all')

    # 🔍 检查 object 类型的非数值列（排除 label）
    non_numeric_cols = chunk.select_dtypes(include=['object']).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col != 'label']
    if non_numeric_cols:
        print(f"⚠️ 删除非数值列: {non_numeric_cols}")
        chunk = chunk.drop(columns=non_numeric_cols)

    # 🏷️ 编码标签列（攻击类为 1，正常为 0）
    chunk['label'] = label_encoder.fit_transform(chunk['label'])

    # ⚠️ 替换 inf/-inf 为 NaN，再统一填 0
    chunk = chunk.replace([float('inf'), float('-inf')], pd.NA).fillna(0)

    # 📊 对除 'label' 和 'protocol' 以外的列进行归一化
    feature_cols = [col for col in chunk.columns if col not in ['label', 'protocol']]
    chunk[feature_cols] = scaler.fit_transform(chunk[feature_cols])

    # 💾 保存处理结果
    output_path = os.path.join(output_dir, f'processed_chunk_{i+1}.csv')
    chunk.to_csv(output_path, index=False)
    print(f"✅ 第 {i+1} 块保存至：{output_path}")

print("\n🎉 所有数据处理完成！")
