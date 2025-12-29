# debug_portmap_loss.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 请替换为你原始数据集中的 Portmap 文件路径
# 建议先测试 03-11/Portmap.csv (如果存在) 或 01-12 下的相关文件
RAW_FILE_PATH = r"E:\CIC-DDoS\CSVS_chart\CSV-03-11\Portmap.csv"


# ===========================================

def analyze_loss(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"正在审计文件: {os.path.basename(file_path)}")
    print("-" * 50)

    # 1. 读取原始数据（不做任何清洗）
    # 使用 chunk 读取以防内存溢出，但为了统计总数，我们先只读列名和标签
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    total_raw = len(df)
    print(f"1. [原始] 总行数: {total_raw}")

    # 2. 检查标签分布
    # 标准化列名
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 查找标签列
    label_col = 'label' if 'label' in df.columns else None
    if not label_col:
        print("⚠️ 未找到 label 列，无法分析标签分布。")
    else:
        print("   原始标签分布:")
        print(df[label_col].value_counts())

        # 只保留 Portmap 攻击
        # 注意：这里要处理大小写，CIC数据集里有时是 'Portmap' 有时是 'Recon-Portmap' 等
        # 我们假设只要不含 benign 且包含 portmap 字眼
        mask_portmap = df[label_col].astype(str).str.contains("Portmap", case=False, na=False)
        df_attack = df[mask_portmap].copy()
        count_attack = len(df_attack)
        print(f"2. [筛选] 仅保留 Portmap 标签后: {count_attack} (损失: {total_raw - count_attack})")

    # 3. 检查 NaN / Infinity
    # 替换 inf 为 nan
    df_attack.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_before_dropna = len(df_attack)
    df_attack.dropna(inplace=True)
    n_after_dropna = len(df_attack)
    print(f"3. [清洗] 剔除 NaN/Inf 后: {n_after_dropna} (损失: {n_before_dropna - n_after_dropna})")

    # 4. 检查重复行 (Duplicates)
    # 剔除无关列再查重 (模拟步骤一的逻辑)
    drop_cols = ['unnamed:_0', 'flow_id', 'source_ip', 'source_port',
                 'destination_ip', 'destination_port', 'timestamp', 'simillarhttp']
    existing_drop = [c for c in drop_cols if c in df_attack.columns]
    df_attack.drop(columns=existing_drop, inplace=True)

    n_before_dedup = len(df_attack)
    df_attack.drop_duplicates(inplace=True)
    n_after_dedup = len(df_attack)

    print(f"4. [去重] 剔除重复行后: {n_after_dedup} (损失: {n_before_dedup - n_after_dedup})")
    print("-" * 50)

    # 结论
    print("📊 最终结论:")
    if n_after_dedup < 5000:
        print(f"   数据量从 {total_raw} 降至 {n_after_dedup} 是经过严格计算的。")
        if (n_before_dedup - n_after_dedup) > (n_before_dropna - n_after_dropna):
            print("   👉 主要原因是：【高度重复】。大部分攻击流量特征完全一致。")
        else:
            print("   👉 主要原因是：【数值异常】。大部分数据包含 Infinity 或 NaN。")
    else:
        print("   数据依然充足，请检查之前的处理脚本是否有其他过滤条件。")


if __name__ == "__main__":
    analyze_loss(RAW_FILE_PATH)