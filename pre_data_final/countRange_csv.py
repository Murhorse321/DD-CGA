# import pandas as pd
#
# # 📂 CSV 文件路径
# csv_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\filtered_dataset.csv'
#
# # 📥 统计总行数
# print("📊 正在统计数据集行数...")
# row_count = sum(1 for _ in open(csv_path)) - 1  # 减去表头行
#
# print(f"✅ 文件总行数（不含表头）：{row_count}")


import pandas as pd

# 数据路径
csv_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\final_balanced_dataset_onehot_Pro.csv'

# 读取数据
df = pd.read_csv(csv_path)

# 获取所有列名
columns = df.columns.tolist()

# 打印列名，每行5个
print("✅ 当前特征列表（每行5个）：")
for i in range(0, len(columns), 5):
    print(", ".join(columns[i:i+5]))

# 输出总特征数
print(f"\n✅ 特征总数（包含标签列）：{len(columns)}")
