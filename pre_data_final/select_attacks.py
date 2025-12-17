import os
import pandas as pd
import random
import re
#挑选与正常流量数据相同的攻击流量
# 💾 路径配置
input_folder = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\processed_chunks'
output_folder = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\attack_flows'
os.makedirs(output_folder, exist_ok=True)

# 📌 目标攻击流量条数
target_attack_count = 6541181
collected_attacks = []
current_total = 0
file_count = 0

# ⏳ 提取数字用于排序
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

# 🔀 打乱顺序后开始处理
all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
random.shuffle(all_files)  # ⚠️ 随机顺序（避免总是按前面文件取）

for filename in all_files:
    if current_total >= target_attack_count:
        break

    file_path = os.path.join(input_folder, filename)
    try:
        df = pd.read_csv(file_path)

        if 'label' not in df.columns:
            continue

        attack_df = df[df['label'] == 1]

        if attack_df.empty:
            continue

        remain = target_attack_count - current_total
        if len(attack_df) > remain:
            attack_df = attack_df.sample(n=remain, random_state=42)

        current_total += len(attack_df)
        file_count += 1

        # 💾 保存攻击流量块
        out_path = os.path.join(output_folder, f'attack_chunk_{file_count}.csv')
        attack_df.to_csv(out_path, index=False)

        print(f"{filename} → ✅ 提取 {len(attack_df)} 条攻击流量，累计: {current_total}/{target_attack_count}")

    except Exception as e:
        print(f"❌ 错误: {filename} -> {e}")

# ✅ 完成
print("\n🎯 攻击流量提取完成")
print(f"📦 生成攻击流量文件数: {file_count}")
print(f"🚨 总攻击流量数: {current_total}")
