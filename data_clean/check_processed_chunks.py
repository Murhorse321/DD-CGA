import os
import pandas as pd
import numpy as np
#检查数据集分成的每块50000条数据是否可用
def standardize_column_names(df):
    # 去除列名前后空格，并统一小写
    df.columns = [col.strip().lower() for col in df.columns]
    return df

folder = 'processed_chunks'
files = sorted(os.listdir(folder))
total_files = len(files)
print(f"📁 共找到 {total_files} 个文件，开始检查...\n")

missing_label_files = []

for f in files:
    path = os.path.join(folder, f)
    df = pd.read_csv(path)

    df = standardize_column_names(df)

    print(f"🔍 正在检查文件：{f}")
    print(f"  📏 行数：{df.shape[0]}，列数：{df.shape[1]}")

    if df.isnull().sum().sum() == 0:
        print("  ✅ 无 NaN 值")
    else:
        print("  ⚠️ 存在 NaN 值")

    if np.isinf(df.select_dtypes(include=[np.number])).values.any():
        print("  ⚠️ 存在 Inf 值")
    else:
        print("  ✅ 无 Inf 值")

    if 'label' in df.columns:
        print("  ✅ 包含 Label 列\n")
    else:
        print("  ❌ 缺少 Label 列\n")
        missing_label_files.append(f)

# 总结
print("📊 检查完成！")
print(f"共缺失 Label 列的文件数量：{len(missing_label_files)}")
if missing_label_files:
    print("如下文件缺失 Label 列：")
    for file in missing_label_files:
        print("  -", file)
else:
    print("✅ 所有文件都包含 Label 列。")
