import os
import pandas as pd
from tqdm import tqdm
#将处理好的合并成块的攻击以及正常流量数据合并成一个文件
# 📂 小块文件夹路径（请替换为你的路径）
chunks_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_chunks'

# 💾 最终合并后的大文件路径
output_csv = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_balanced_dataset.csv'

# 🔍 获取所有小块文件（按顺序排序）
chunk_files = sorted([os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith('.csv')])

print(f"📦 共发现 {len(chunk_files)} 个小块文件，开始合并...")

# ✅ 初始化标签统计字典
label_counts = {0: 0, 1: 0}
total_records = 0
first_chunk = True

# 🔁 合并过程（带进度条）
for chunk_file in tqdm(chunk_files, desc="🔄 合并进度"):
    chunk_df = pd.read_csv(chunk_file)

    # 统计标签数量
    label_counts[0] += (chunk_df['label'] == 0).sum()
    label_counts[1] += (chunk_df['label'] == 1).sum()

    # 统计总记录数
    total_records += len(chunk_df)

    # 追加写入
    chunk_df.to_csv(output_csv, mode='a', index=False, header=first_chunk)
    first_chunk = False

print(f"\n✅ 合并完成，已保存至：{output_csv}")
print(f"📊 合并后标签统计：正常流量（label=0）：{label_counts[0]} | 攻击流量（label=1）：{label_counts[1]}")
print(f"📈 合并后总记录数：{total_records}")

# ✅ 数据是否平衡自动检测
if label_counts[0] == label_counts[1]:
    print("✅ 数据集平衡！")
else:
    print("⚠️ 数据集不平衡，请检查数据采样过程！")
