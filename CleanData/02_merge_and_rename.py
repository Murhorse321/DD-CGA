# 02_merge_and_rename.py
# -*- coding: utf-8 -*-

import os
import shutil
from tqdm import tqdm

# ================= 配置区域 =================
# 这里请填入你“步骤一”清洗后的两个输出文件夹路径
# 如果你之前都输出到了同一个文件夹且确定没有同名覆盖，可以将两个变量指向同一个路径
# 但建议最好是分开的路径以确保安全

# 示例：假设你把第一天的数据洗好放在了 data/step1_cleaned_0112
# 第二天的数据放在了 data/step1_cleaned_0311
DIR_DAY1 = r"D:\Desktop\C_G_A\CNN_GRU_ATTENTION\CleanData\data\step1_cleaned"  # 请修改为实际路径 (01-12)
DIR_DAY2 = r"D:\Desktop\C_G_A\CNN_GRU_ATTENTION\CleanData\data\step1_cleaned_1"  # 请修改为实际路径 (03-11)

# 新的合并输出目录
OUTPUT_DIR = "data/step2_merged"


# ===========================================

def merge_files(src_dir, prefix, output_dir):
    if not os.path.exists(src_dir):
        print(f"⚠️ 警告: 找不到源目录 {src_dir}，跳过该部分。")
        return 0

    files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
    count = 0

    print(f"正在处理目录: {src_dir} (前缀: {prefix})")
    for filename in tqdm(files):
        src_path = os.path.join(src_dir, filename)

        # 构造新文件名：前缀 + 原文件名
        # 例如: 01-12_DrDoS_DNS.csv
        new_filename = f"{prefix}_{filename}"
        dst_path = os.path.join(output_dir, new_filename)

        # 复制文件 (使用 copy2 保留元数据，或者 copyfile 仅复制内容)
        # 这里使用 move 还是 copy？为了安全，建议使用 copy，保留上一名为备份
        shutil.copyfile(src_path, dst_path)
        count += 1

    return count


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    else:
        print(f"输出目录已存在: {OUTPUT_DIR} (可能会覆盖同名文件)")

    print("开始合并与重命名...")
    print("-" * 50)

    # 处理 Day 1 (01-12)
    c1 = merge_files(DIR_DAY1, "01-12", OUTPUT_DIR)

    # 处理 Day 2 (03-11)
    c2 = merge_files(DIR_DAY2, "03-11", OUTPUT_DIR)

    print("-" * 50)
    print(f"✅ 合并完成！")
    print(f"  - 01-12 (Training Day) 文件数: {c1}")
    print(f"  - 03-11 (Testing Day)  文件数: {c2}")
    print(f"  - 总文件数: {c1 + c2}")
    print(f"📂 所有文件已汇总至: {OUTPUT_DIR}")

    # 简单的完整性检查
    all_files = os.listdir(OUTPUT_DIR)
    print(f"当前合并目录下文件列表 ({len(all_files)} 个):")
    # 只打印前5个和后5个避免刷屏
    if len(all_files) > 10:
        print(all_files[:5], "...", all_files[-5:])
    else:
        print(all_files)

    print("\n请确认文件数量无误后，回复“可以继续”进入标签编码步骤。")


if __name__ == "__main__":
    main()