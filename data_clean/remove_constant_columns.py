# import pandas as pd
#
# # === 修改为你的整合后文件路径 ===
# input_file = "balanced_dataset.csv"
# output_file = "balanced_dataset_no_constant.csv"
#
# print("📥 正在加载数据文件...")
# df = pd.read_csv(input_file)
#
# print(f"📊 原始数据维度：{df.shape[0]} 行, {df.shape[1]} 列")
#
# # === 检查恒定列 ===
# print("🔍 正在识别恒定列...")
# constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
#
# # === 输出恒定列名 ===
# if constant_columns:
#     print(f"🧹 检测到 {len(constant_columns)} 个恒定列，将删除以下列：")
#     for col in constant_columns:
#         print(f"   - {col}")
# else:
#     print("✅ 未发现恒定列。")
#
# # === 删除恒定列 ===
# df_cleaned = df.drop(columns=constant_columns)
#
# # === 保存处理后的文件 ===
# df_cleaned.to_csv(output_file, index=False)
# print(f"\n✅ 清洗后的数据已保存至：{output_file}")
# print(f"📐 新数据维度：{df_cleaned.shape[0]} 行, {df_cleaned.shape[1]} 列")



import pandas as pd

# === 修改为你的整合后文件路径 ===
input_file = "balanced_dataset_test_cnn.csv"
output_file = "balanced_dataset_test_cnn_no_constant.csv"

print("📥 正在加载数据文件...")
df = pd.read_csv(input_file)

print(f"📊 原始数据维度：{df.shape[0]} 行, {df.shape[1]} 列")

# === 检查恒定列 ===
print("🔍 正在识别恒定列...")
constant_columns = [col for col in df.columns if df[col].nunique() <= 1]

# === 输出恒定列名 ===
if constant_columns:
    print(f"🧹 检测到 {len(constant_columns)} 个恒定列，将删除以下列：")
    for col in constant_columns:
        print(f"   - {col}")
else:
    print("✅ 未发现恒定列。")

# === 删除恒定列 ===
df_cleaned = df.drop(columns=constant_columns)

# === 保存处理后的文件 ===
df_cleaned.to_csv(output_file, index=False)
print(f"\n✅ 清洗后的数据已保存至：{output_file}")
print(f"📐 新数据维度：{df_cleaned.shape[0]} 行, {df_cleaned.shape[1]} 列")
