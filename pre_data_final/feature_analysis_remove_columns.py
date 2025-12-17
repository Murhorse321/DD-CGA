import pandas as pd

# 数据路径
csv_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_balanced_dataset_onehot_Pro.csv'
save_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\filtered_dataset.csv'

# 读取数据
df = pd.read_csv(csv_path)

# 定义要删除的列（已移除 protocol_6 和 protocol_17，仅删除 protocol_0）
remove_columns = [
    'fwd psh flags',
    'syn flag count',
    'rst flag count',
    'ack flag count',
    'urg flag count',
    'cwe flag count',
    'protocol_0'  # 仅删除 protocol_0，保留 protocol_6 和 protocol_17
]

# 确认删除前的列
original_columns = df.columns.tolist()

# 执行删除
df_filtered = df.drop(columns=remove_columns)

# 确认删除后的列
remaining_columns = df_filtered.columns.tolist()

# 统计信息
print(f"✅ 删除的特征：{remove_columns}")
print(f"✅ 删除前特征总数（包含标签列）：{len(original_columns)}")
print(f"✅ 删除后特征总数（包含标签列）：{len(remaining_columns)}")
print(f"✅ 剩余特征：{remaining_columns}")

# 保存新数据集
df_filtered.to_csv(save_path, index=False)
print(f"✅ 新数据集已保存至：{save_path}")
