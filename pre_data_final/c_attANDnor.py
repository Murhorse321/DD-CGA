import pandas as pd
import os
#从提取到的分块的正常流量以及攻击流量中随机打乱并合并，输出每个大小200000的块
# 📁 输入文件路径（请修改为你的路径）
normal_csv = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\combine_normal.csv'
attack_csv = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\combine_attack.csv'

# 📁 输出路径
output_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_chunks'
os.makedirs(output_dir, exist_ok=True)

# ⚙️ 分块大小（每次加载10万条）
chunk_size = 200000

# 统计总条数
normal_total = sum(1 for _ in open(normal_csv)) - 1
attack_total = sum(1 for _ in open(attack_csv)) - 1

# 计算循环次数
normal_iter = pd.read_csv(normal_csv, chunksize=chunk_size)
attack_iter = pd.read_csv(attack_csv, chunksize=chunk_size)

chunk_id = 1
print("🚀 开始分块读取、打乱并保存...")

for normal_chunk, attack_chunk in zip(normal_iter, attack_iter):
    # 保证两块数据大小一致
    min_len = min(len(normal_chunk), len(attack_chunk))
    normal_chunk = normal_chunk.sample(n=min_len, random_state=42).reset_index(drop=True)
    attack_chunk = attack_chunk.sample(n=min_len, random_state=42).reset_index(drop=True)

    # 合并 + 打乱
    combined = pd.concat([normal_chunk, attack_chunk], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 保存分块文件
    output_path = os.path.join(output_dir, f'balanced_chunk_{chunk_id}.csv')
    combined.to_csv(output_path, index=False)
    print(f"✅ 已保存：{output_path}，共 {len(combined)} 条数据")
    chunk_id += 1

print("🎉 所有数据已分块合并并打乱完成！")
