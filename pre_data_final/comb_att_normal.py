import os
import pandas as pd

# 📁 输入目录路径（请修改为你实际的路径）
normal_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\normal_flows'
attack_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\attack_flows'
#将之前提取到的攻击以及正常流量进行合并（提取到的不规则大小的流量块）
# 💾 输出文件路径
normal_output = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\combine_normal.csv'
attack_output = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\combine_attack.csv'

# 🧱 合并函数
def combine_csv_files(input_dir, output_path, description):
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    print(f"📦 正在合并 {description} 文件，共计 {len(all_files)} 个文件...")

    df_list = []
    for f in all_files:
        file_path = os.path.join(input_dir, f)
        df = pd.read_csv(file_path)
        df_list.append(df)
        print(f"✅ 加载完成: {f}，包含 {len(df)} 条数据")

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"🎯 合并完成并保存至：{output_path}，总条数：{len(combined_df)}")

# 🚀 执行合并
combine_csv_files(normal_dir, normal_output, "正常流量")
combine_csv_files(attack_dir, attack_output, "攻击流量")
