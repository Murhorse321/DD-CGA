import pandas as pd

# 读取数据
df = pd.read_csv("D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\select_sample_2M.csv")

# 去掉标签列
features = df.drop(columns=["label"])

# 查看每个特征的均值和标准差
print(features.mean())
print(features.std())
