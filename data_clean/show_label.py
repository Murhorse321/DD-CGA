# import pandas as pd
#
# file_path = 'D:\Desktop\CIC-DDoS\CSVs\CICDDoS2019_Merged.csv'  # 按你真实路径修改
#
# # 读取前5行看看列名
# df_sample = pd.read_csv(file_path, nrows=5)
# print("📋 列名如下：")
# print(df_sample.columns.tolist())


# import pandas as pd
#
# file_path = 'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\cleaned_BRS.csv'  # 按你真实路径修改
#
# # 读取前5行看看列名
# df_sample = pd.read_csv(file_path, nrows=5)
# print("📋 列名如下：")
# print(df_sample.columns.tolist())



# import pandas as pd
#
# file_path = 'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\Portmap.csv'  # 按你真实路径修改
#
# # 读取前5行看看列名
# df_sample = pd.read_csv(file_path, nrows=5)
# print("📋 列名如下：")
# print(df_sample.columns.tolist())

# import pandas as pd
#
# file_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\cleaned_BRS.csv'  # 使用原始字符串
#
# # 读取前5行看看列名
# df_sample = pd.read_csv(file_path, nrows=5)
# columns = df_sample.columns.tolist()
#
# print("📋 标签列表（每行显示5个）：")
# # 每5个标签为一组显示
# for i in range(0, len(columns), 5):
#     # 获取当前组的5个标签（或剩余的标签）
#     group = columns[i:i+5]
#     # 创建带编号的标签字符串
#     numbered_columns = [f"{i+j+1}. '{col}'" for j, col in enumerate(group)]
#     # 将组内标签合并为一个字符串，用制表符分隔
#     print("\t".join(numbered_columns))

import pandas as pd

file_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\Portmap.csv'  # 使用原始字符串

# 读取前5行看看列名
df_sample = pd.read_csv(file_path, nrows=5)
columns = df_sample.columns.tolist()

print("📋 标签列表（每行显示5个）：")
# 每5个标签为一组显示
for i in range(0, len(columns), 5):
    # 获取当前组的5个标签（或剩余的标签）
    group = columns[i:i+5]
    # 创建带编号的标签字符串
    numbered_columns = [f"{i+j+1}. '{col}'" for j, col in enumerate(group)]
    # 将组内标签合并为一个字符串，用制表符分隔
    print("\t".join(numbered_columns))