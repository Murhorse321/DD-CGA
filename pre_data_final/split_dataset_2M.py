import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取数据
data_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\select_sample_2M.csv'
df = pd.read_csv(data_path)

# 2. 特征和标签分开
X = df.drop(columns=["label"])
y = df["label"]

# 3. 先划分训练集 (70%) 和临时集 (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4. 再从临时集划分验证集 (15%) 和测试集 (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 5. 合并特征和标签，存储为 csv
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("val_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

print("数据集划分完成 ✅")
print(f"训练集: {train_df.shape[0]} 行")
print(f"验证集: {val_df.shape[0]} 行")
print(f"测试集: {test_df.shape[0]} 行")
