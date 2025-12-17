import pandas as pd
#删除重复标签、不适合建模的标签、以及恒定是零的标签，
# #📊 数据加载完成，总记录数：13082362，总特征数（包含标签）：79
#
# 🗑️ 已删除不适合建模的列：['fwd header length.1']
# 📊 删除后剩余特征数（包含标签）：78
#
# 🗑️ 已删除恒定列（数值不变）：['bwd psh flags', 'fwd urg flags', 'bwd urg flags', 'fin flag count',
# 'psh flag count', 'ece flag count', 'fwd avg bytes/bulk', 'fwd avg packets/bulk',
# 'fwd avg bulk rate', 'bwd avg bytes/bulk', 'bwd avg packets/bulk', 'bwd avg bulk rate']
# 📊 删除恒定列后剩余特征数（包含标签）：66
#
# ✅ 最终保留列（共 66 个，包括标签列）：
# ['protocol', 'flow duration', 'total fwd packets', 'total backward packets',
# 'total length of fwd packets', 'total length of bwd packets', 'fwd packet length max',
# 'fwd packet length min', 'fwd packet length mean', 'fwd packet length std',
# 'bwd packet length max', 'bwd packet length min', 'bwd packet length mean',
# 'bwd packet length std', 'flow bytes/s', 'flow packets/s', 'flow iat mean',
# 'flow iat std', 'flow iat max', 'flow iat min', 'fwd iat total', 'fwd iat mean',
# 'fwd iat std', 'fwd iat max', 'fwd iat min', 'bwd iat total', 'bwd iat mean',
# 'bwd iat std', 'bwd iat max', 'bwd iat min', 'fwd psh flags', 'fwd header length',
# 'bwd header length', 'fwd packets/s', 'bwd packets/s', 'min packet length',
# 'max packet length', 'packet length mean', 'packet length std', 'packet length variance',
# 'syn flag count', 'rst flag count', 'ack flag count', 'urg flag count', 'cwe flag count',
# 'down/up ratio', 'average packet size', 'avg fwd segment size', 'avg bwd segment size',
# 'subflow fwd packets', 'subflow fwd bytes', 'subflow bwd packets', 'subflow bwd bytes',
# 'init_win_bytes_forward', 'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
# 'active mean', 'active std', 'active max', 'active min', 'idle mean', 'idle std', 'idle max',
# 'idle min', 'label']

# 📂 输入路径（你已合并的数据集）
csv_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_balanced_dataset.csv'

# 💾 输出路径
output_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_balanced_dataset_cleaned.csv'

# 读取数据
print("📥 正在加载数据...")
df = pd.read_csv(csv_path)

print(f"📊 数据加载完成，总记录数：{len(df)}，总特征数（包含标签）：{df.shape[1]}")

# ==================== 步骤一：删除不适合建模的列 ==================== #
# 📝 手动指定不适合建模的列
# 如果你知道当初没删干净的非数值列/标识列，可以加在这里
unwanted_columns = [
    'Unnamed: 0', 'flow id', 'source ip', 'source port',
    'destination ip', 'destination port', 'timestamp',
    'simillarhttp', 'inbound', 'fwd header length.1'
]

# 自动匹配存在的列删除
unwanted_columns = [col.lower().strip() for col in unwanted_columns]
df.columns = df.columns.str.strip().str.lower()
cols_to_delete = [col for col in unwanted_columns if col in df.columns]
df.drop(columns=cols_to_delete, inplace=True)

print(f"\n🗑️ 已删除不适合建模的列：{cols_to_delete}")
print(f"📊 删除后剩余特征数（包含标签）：{df.shape[1]}")

# ==================== 步骤二：删除恒定列 ==================== #
constant_cols = [col for col in df.columns if df[col].nunique() == 1 and col != 'label']

if constant_cols:
    df.drop(columns=constant_cols, inplace=True)
    print(f"\n🗑️ 已删除恒定列（数值不变）：{constant_cols}")
else:
    print("\n✅ 未检测到恒定列。")

print(f"📊 删除恒定列后剩余特征数（包含标签）：{df.shape[1]}")

# ==================== 输出最终保留列 ==================== #
remaining_columns = df.columns.tolist()
print(f"\n✅ 最终保留列（共 {len(remaining_columns)} 个，包括标签列）：")
print(remaining_columns)

# ==================== 保存处理后的数据 ==================== #
df.to_csv(output_path, index=False)
print(f"\n💾 已保存清洗后的数据集至：{output_path}")
