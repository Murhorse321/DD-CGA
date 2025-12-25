import os
import pandas as pd
from collections import Counter
#统计每种流量共有多少条数据
# 设置参数
chunk_dir = 'processed_chunks'
label_col = 'label'  # 标签列必须为小写
print_interval = 50  # 每处理几个块打印一次

# 初始化计数器
total_counts = Counter()
processed_files = 0

# 获取所有 CSV 块文件
chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.csv')])

print(f"🔍 开始统计 {len(chunk_files)} 个块中的样本数量...\n")

for file_name in chunk_files:
    file_path = os.path.join(chunk_dir, file_name)
    df = pd.read_csv(file_path)

    # 如果没有标签列，则跳过
    if label_col not in df.columns:
        continue

    # 更新标签数量计数
    label_counts = df[label_col].value_counts().to_dict()
    total_counts.update(label_counts)

    processed_files += 1

    # 每 print_interval 个块输出一次当前结果
    if processed_files % print_interval == 0:
        print(f"📦 已处理 {processed_files} 个块")
        print(f"   ✅ 正常流量（label=0）：{total_counts.get(0, 0)}")
        print(f"   ⚠️ 攻击流量（label=1）：{total_counts.get(1, 0)}\n")

# 最终输出统计结果
print("🎉 所有块统计完成！")
print(f"✅ 正常流量（label=0）：{total_counts.get(0, 0)} 条")
print(f"⚠️ 攻击流量（label=1）：{total_counts.get(1, 0)} 条")
print(f"📊 总样本数：{sum(total_counts.values())} 条")
