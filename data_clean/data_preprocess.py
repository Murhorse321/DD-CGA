import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#数据预处理——删掉null 以及其他不合法的值
# 📍 文件路径（请根据你文件的位置修改）
file_path = 'E:\CIC-DDoS\CSVs\CICDDoS2019_Merged.csv'

# 📁 输出目录
output_dir = 'processed_chunks'
os.makedirs(output_dir, exist_ok=True)

# ⚙️ 设置分块大小
chunk_size = 50000

# 🚫 无用列列表（可根据实际进一步精简）
useless_columns = [
    'flow id', 'source ip', 'source port', 'destination ip', 'destination port',
    'timestamp', 'simillarhttp', 'inbound', 'unnamed: 0'
]

# ✅ 初始化工具
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

print("📥 正在读取并分块处理数据...")

for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, encoding='utf-8')):
    print(f"\n📦 正在处理第 {i+1} 块数据...")

    # ✂️ 清洗列名：去空格 + 小写
    chunk.columns = chunk.columns.str.strip().str.lower()

    # 🕵️‍♂️ 检查标签列
    if 'label' not in chunk.columns:
        print(f"⚠️ 跳过第 {i+1} 块，未找到 label 列。")
        continue

    # 🚮 删除无用列（如果存在）
    chunk = chunk.drop(columns=[col for col in useless_columns if col in chunk.columns], errors='ignore')

    # 🧹 删除全为NaN的列（可能某些列在某些chunk中为空）
    chunk = chunk.dropna(axis=1, how='all')

    # 🔎 检查是否仍包含非数值特征
    non_numeric_cols = chunk.select_dtypes(include=['object']).columns.tolist()
    non_numeric_cols = [col for col in non_numeric_cols if col != 'label']

    # 🚫 若存在其他非数值列，先删除（如协议类型）
    if non_numeric_cols:
        print(f"⚠️ 删除非数值列: {non_numeric_cols}")
        chunk = chunk.drop(columns=non_numeric_cols)

    # 🏷️ 编码标签列（攻击 -> 1，正常 -> 0）
    chunk['label'] = label_encoder.fit_transform(chunk['label'])

    # ⚠️ 替换 inf 和 -inf 为 NaN，再统一填 0
    chunk = chunk.replace([float('inf'), float('-inf')], pd.NA)
    chunk = chunk.fillna(0)

    # 📊 特征归一化（除了标签列）

    feature_cols = chunk.columns[chunk.columns != 'label']
    chunk[feature_cols] = scaler.fit_transform(chunk[feature_cols])

    # 💾 保存当前处理块
    output_path = os.path.join(output_dir, f'processed_chunk_{i+1}.csv')
    chunk.to_csv(output_path, index=False)
    print(f"✅ 第 {i+1} 块处理完成并保存至：{output_path}")

print("\n🎉 所有数据处理完成！")
