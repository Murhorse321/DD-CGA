import pandas as pd

# 数据路径
csv_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\filtered_dataset.csv'
save_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\datas\select_sample_2M.csv'

# 读取数据
df = pd.read_csv(csv_path)

# 确认标签列名称
label_col = 'label'

# 分别筛选
df_normal = df[df[label_col] == 0].sample(n=1000000, random_state=42)
df_attack = df[df[label_col] == 1].sample(n=1000000, random_state=42)

# 合并 + 打乱
df_balanced = pd.concat([df_normal, df_attack]).sample(frac=1.0, random_state=42).reset_index(drop=True)

# 保存
df_balanced.to_csv(save_path, index=False)

print(f"✅ 已从数据集中各随机抽取 100 万条正常流量与攻击流量，总计 {len(df_balanced)} 条，已保存至：{save_path}")
