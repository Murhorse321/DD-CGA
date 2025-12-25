# import pandas as pd
# import numpy as np
#
# # 读取数据
# df = pd.read_csv("balanced_dataset_no_constant.csv")
#
# # 1️⃣ 删除重复列
# if 'fwd header length.1' in df.columns:
#     df.drop(columns=['fwd header length.1'], inplace=True)
#
# # 2️⃣ 删除稀疏列（如 syn flag count）
# if 'syn flag count' in df.columns and df['syn flag count'].sum() < 100:
#     df.drop(columns=['syn flag count'], inplace=True)
#
# # # 3️⃣ 处理 protocol 列：使用 one-hot 编码，替代原始 protocol
# # if 'protocol' in df.columns:
# #     protocol_dummies = pd.get_dummies(df['protocol'], prefix="protocol")
# #     df.drop(columns=['protocol'], inplace=True)
# #     df = pd.concat([df, protocol_dummies], axis=1)
#
# # 4️⃣ 替换 inf/-inf 为 NaN，并统一填 0
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.fillna(0, inplace=True)
#
# # 5️⃣ label 列除外，将所有列转换为 float32 类型（防止字符串/整数混入）
# for col in df.columns:
#     if col != 'label':
#         df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
#
# # ✅ 最终确认
# print(f"✅ 清洗完成，当前特征列数（不含 label）：{df.shape[1] - 1}")
# print(f"当前所有列名：{df.columns.tolist()}")
# df.to_csv("cleaned_BRS.csv", index=False)


import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("balanced_dataset_test_cnn_no_constant.csv")

# 1️⃣ 删除重复列
if 'fwd header length.1' in df.columns:
    df.drop(columns=['fwd header length.1'], inplace=True)

# 2️⃣ 删除稀疏列（如 syn flag count）
if 'syn flag count' in df.columns and df['syn flag count'].sum() < 100:
    df.drop(columns=['syn flag count'], inplace=True)

# # 3️⃣ 处理 protocol 列：使用 one-hot 编码，替代原始 protocol
# if 'protocol' in df.columns:
#     protocol_dummies = pd.get_dummies(df['protocol'], prefix="protocol")
#     df.drop(columns=['protocol'], inplace=True)
#     df = pd.concat([df, protocol_dummies], axis=1)

# 4️⃣ 替换 inf/-inf 为 NaN，并统一填 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 5️⃣ label 列除外，将所有列转换为 float32 类型（防止字符串/整数混入）
for col in df.columns:
    if col != 'label':
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

# ✅ 最终确认
print(f"✅ 清洗完成，当前特征列数（不含 label）：{df.shape[1] - 1}")
print(f"当前所有列名：{df.columns.tolist()}")
df.to_csv("cnn_ddos_test.csv", index=False)
