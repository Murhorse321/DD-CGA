import os
import pandas as pd
#选出所有的正常流量
# ✅ 替换为你实际的路径
input_folder = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\processed_chunks'
output_folder = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\normal_flows'

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

# 初始化统计
total_normal = 0
total_attack = 0
file_count = 0

# 遍历每个CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(file_path)

            if 'label' not in df.columns:
                print(f"⚠️ 跳过 {filename}（无 label 列）")
                continue

            # 统计
            normal_df = df[df['label'] == 0]
            attack_df = df[df['label'] == 1]
            normal_count = len(normal_df)
            attack_count = len(attack_df)

            # 累加统计
            total_normal += normal_count
            total_attack += attack_count
            file_count += 1

            print(f"{filename} → ✅ 正常流量: {normal_count}, 🚨 攻击流量: {attack_count}")

            # 保存正常流量数据
            if normal_count > 0:
                output_path = os.path.join(output_folder, f'normal_chunk_{file_count}.csv')
                normal_df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"❌ 读取文件出错: {filename}，错误: {e}")

# 最终统计
print("\n📊 总结：")
print(f"处理文件总数：{file_count}")
print(f"正常流量总数（label=0）：{total_normal}")
print(f"攻击流量总数（label=1）：{total_attack}")
print(f"总记录数：{total_normal + total_attack}")
