# import os
# import pandas as pd
# import random
#
# # 配置参数
# #平衡随机抽样BRS
# chunk_dir = 'processed_chunks'
# target_per_class = 250000
# output_file = 'balanced_dataset.csv'
#
# # 初始化统计
# normal_samples = []
# attack_samples = []
# normal_count = 0
# attack_count = 0
#
# # 随机打乱块顺序
# chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.csv')])
# random.shuffle(chunk_files)
#
# print(f"📦 总共有 {len(chunk_files)} 个数据块，开始执行平衡抽样...\n")
#
# for idx, file in enumerate(chunk_files):
#     path = os.path.join(chunk_dir, file)
#     df = pd.read_csv(path)
#
#     if 'label' not in df.columns:
#         print(f"⚠️ {file} 中未找到 label 列，跳过该块")
#         continue
#
#     # 分离两类
#     normal_df = df[df['label'] == 0]
#     attack_df = df[df['label'] == 1]
#
#     # 按需采样（若剩余目标不足当前块数，按需采样）
#     if normal_count < target_per_class:
#         need_n = min(target_per_class - normal_count, len(normal_df))
#         normal_samples.append(normal_df.sample(n=need_n, random_state=42))
#         normal_count += need_n
#
#     if attack_count < target_per_class:
#         need_a = min(target_per_class - attack_count, len(attack_df))
#         attack_samples.append(attack_df.sample(n=need_a, random_state=42))
#         attack_count += need_a
#
#     print(f"✅ 处理 {file}：累计 正常流量={normal_count}，攻击流量={attack_count}")
#
#     if normal_count >= target_per_class and attack_count >= target_per_class:
#         print("\n🎉 已采集足够样本，结束遍历。\n")
#         break
#
# # 合并并保存
# final_df = pd.concat(normal_samples + attack_samples, ignore_index=True)
# final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
# final_df.to_csv(output_file, index=False)
#
# print(f"✅ 已成功保存平衡数据至：{output_file}")
# print(f"🔢 最终样本数量：{len(final_df)}（每类 {target_per_class} 条）")


import os
import pandas as pd
import random

# 配置参数
#平衡随机抽样BRS
chunk_dir = 'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\processed_chunks'
target_per_class = 50000
output_file = 'balanced_dataset_test_cnn.csv'

# 初始化统计
normal_samples = []
attack_samples = []
normal_count = 0
attack_count = 0

# 随机打乱块顺序
chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.csv')])
random.shuffle(chunk_files)

print(f"📦 总共有 {len(chunk_files)} 个数据块，开始执行平衡抽样...\n")

for idx, file in enumerate(chunk_files):
    path = os.path.join(chunk_dir, file)
    df = pd.read_csv(path)

    if 'label' not in df.columns:
        print(f"⚠️ {file} 中未找到 label 列，跳过该块")
        continue

    # 分离两类
    normal_df = df[df['label'] == 0]
    attack_df = df[df['label'] == 1]

    # 按需采样（若剩余目标不足当前块数，按需采样）
    if normal_count < target_per_class:
        need_n = min(target_per_class - normal_count, len(normal_df))
        normal_samples.append(normal_df.sample(n=need_n, random_state=42))
        normal_count += need_n

    if attack_count < target_per_class:
        need_a = min(target_per_class - attack_count, len(attack_df))
        attack_samples.append(attack_df.sample(n=need_a, random_state=42))
        attack_count += need_a

    print(f"✅ 处理 {file}：累计 正常流量={normal_count}，攻击流量={attack_count}")

    if normal_count >= target_per_class and attack_count >= target_per_class:
        print("\n🎉 已采集足够样本，结束遍历。\n")
        break

# 合并并保存
final_df = pd.concat(normal_samples + attack_samples, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
final_df.to_csv(output_file, index=False)

print(f"✅ 已成功保存平衡数据至：{output_file}")
print(f"🔢 最终样本数量：{len(final_df)}（每类 {target_per_class} 条）")
