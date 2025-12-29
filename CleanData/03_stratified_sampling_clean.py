# 03_stratified_sampling_clean.py
# -*- coding: utf-8 -*-

import os
import glob
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================= 🧪 实验配置区域 =================
INPUT_DIR = "data/step2_merged"
OUTPUT_FILE = "data/step3_balanced.csv"

# 【目标配额】
TARGET_SAMPLES_PER_CLASS = 10000

RANDOM_SEED = 42
CHUNK_SIZE = 100000

# 【核心 1：黑名单】
# 显式剔除噪声类别，保证数据集纯净
IGNORE_LABELS = [
    'WebDDoS',  # 样本过少 (<500)，属于噪声
     # 如果你不想合并到 UDP-lag，也可以剔除，但通常建议保留并合并
    # 如果发现其他只有几百条的怪异类别，也可以加在这里
]

# 【核心 2：类别映射字典】
# 将同源攻击归一化
ATTACK_MAPPING = {
    # UDP 家族
    'DrDoS_UDP': 'UDP',
    'UDP': 'UDP',

    # LDAP 家族
    'DrDoS_LDAP': 'LDAP',
    'LDAP': 'LDAP',

    # MSSQL 家族
    'DrDoS_MSSQL': 'MSSQL',
    'MSSQL': 'MSSQL',

    # NetBIOS 家族
    'DrDoS_NetBIOS': 'NetBIOS',
    'NetBIOS': 'NetBIOS',

    # Syn 家族
    'Syn': 'Syn',

    # UDP-Lag 拼写修正 (注意：Syn 和 UDP-lag 是不同的)
    'UDP-lag': 'UDPLag',
    'UDPLag': 'UDPLag',
}


# =================================================

def normalize_label(label):
    return str(label).strip()


def get_unified_class(filename):
    """文件名 -> 统一类别"""
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]

    # 去除日期前缀
    raw_name = name_no_ext
    for prefix in ["01-12_", "03-11_"]:
        if raw_name.startswith(prefix):
            raw_name = raw_name.replace(prefix, "")
            break

    # 映射
    return ATTACK_MAPPING.get(raw_name, raw_name)


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 找不到目录 {INPUT_DIR}")
        return

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print("❌ 未找到 CSV 文件。")
        return

    # 1. 扫描与归类
    print("🔍 正在扫描文件并建立索引...")
    class_file_map = {}

    for f in csv_files:
        unified_class = get_unified_class(f)

        # 🚫 检查是否在黑名单中
        if unified_class in IGNORE_LABELS:
            print(f"  ⚠️ 跳过被忽略的类别文件: {unified_class} ({os.path.basename(f)})")
            continue

        if unified_class not in class_file_map:
            class_file_map[unified_class] = []
        class_file_map[unified_class].append(f)

    print("-" * 40)
    print(f"✅ 最终纳入采样的攻击类别 ({len(class_file_map)} 类):")
    for k, v in class_file_map.items():
        print(f"     [{k}]: {len(v)} 个源文件")
    print("-" * 40)

    final_dfs = []

    # 2.1 提取良性 (Benign)
    print("\n📦 [1/2] 正在提取良性流量 (Benign)...")
    total_benign = 0
    for f in tqdm(csv_files, desc="Scanning Benign"):
        # 即使是被忽略的 WebDDoS 文件，里面也可能有 Benign，所以都要扫一遍
        try:
            chunks = []
            with pd.read_csv(f, chunksize=CHUNK_SIZE) as reader:
                for chunk in reader:
                    if 'label' not in chunk.columns: continue
                    chunk['label'] = chunk['label'].apply(normalize_label)

                    # 提取 Benign
                    benign_part = chunk[chunk['label'].str.lower() == 'benign'].copy()
                    if not benign_part.empty:
                        benign_part['label'] = 'Benign'
                        chunks.append(benign_part)

            if chunks:
                df_b = pd.concat(chunks)
                final_dfs.append(df_b)
                total_benign += len(df_b)
        except Exception as e:
            print(f"  ⚠️ 读取 {os.path.basename(f)} 失败: {e}")

    print(f"  -> ✅ 良性样本总数: {total_benign}")

    # 2.2 提取攻击 (Attack)
    print(f"\n📦 [2/2] 正在提取攻击流量 (Target={TARGET_SAMPLES_PER_CLASS}/类)...")

    for atk_class, file_list in class_file_map.items():
        num_files = len(file_list)
        quota_per_file = math.ceil(TARGET_SAMPLES_PER_CLASS / num_files)

        print(f"  -> 处理: {atk_class} (每文件限额 {quota_per_file})")

        class_collected = 0

        for f in file_list:
            file_collected_df = []
            with pd.read_csv(f, chunksize=CHUNK_SIZE) as reader:
                for chunk in reader:
                    if 'label' not in chunk.columns: continue
                    chunk['label'] = chunk['label'].apply(normalize_label)

                    # 剔除 Benign 和 黑名单label (双重保险)
                    # 有时候 WebDDoS.csv 里不仅有 Benign 还有 WebDDoS 标签
                    mask_atk = (chunk['label'].str.lower() != 'benign') & \
                               (~chunk['label'].isin(IGNORE_LABELS))

                    df_atk = chunk[mask_atk]
                    if df_atk.empty: continue

                    file_collected_df.append(df_atk)

            if not file_collected_df:
                continue

            df_file_total = pd.concat(file_collected_df, ignore_index=True)

            # 抽样
            if len(df_file_total) > quota_per_file:
                df_sampled = df_file_total.sample(n=quota_per_file, random_state=RANDOM_SEED)
            else:
                df_sampled = df_file_total

            # 重命名为统一标签
            df_sampled = df_sampled.copy()
            df_sampled['label'] = atk_class

            final_dfs.append(df_sampled)
            class_collected += len(df_sampled)

        print(f"     ✅ 已收集 {atk_class}: {class_collected} 条")

    # 3. 保存
    if not final_dfs:
        print("❌ 未提取到数据")
        return

    print("\n🔄 合并与打乱...")
    full_df = pd.concat(final_dfs, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    full_df['label_int'] = full_df['label'].apply(lambda x: 0 if x == 'Benign' else 1)

    print(f"💾 保存至 {OUTPUT_FILE} ...")
    full_df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print("🎉 完美数据集构建完成！")
    print(full_df['label'].value_counts())
    print("-" * 50)


if __name__ == "__main__":
    main()