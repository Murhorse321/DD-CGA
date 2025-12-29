# 03_clean_structural.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

INPUT_FILE = "data/step3_balanced.csv"
OUTPUT_FILE = "data/step3_struct_cleaned.csv"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        return

    print("🚀 开始结构性清洗 (去除恒定列/全空列)...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   原始维度: {df.shape}")

    # 1. 剔除全空列
    df.dropna(axis=1, how='all', inplace=True)

    # 2. 剔除单值列 (方差为0)
    # 仅检查数值列，避开 label 字符串
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 保护 label_int 不被误删
    cols_to_check = [c for c in numeric_cols if c != 'label_int']

    # 找到标准差为 0 的列
    const_cols = [c for c in cols_to_check if df[c].std() == 0]

    if const_cols:
        df.drop(columns=const_cols, inplace=True)
        print(f"   -> 剔除恒定列: {len(const_cols)} 个")
        print(f"      例如: {const_cols[:5]}...")
    else:
        print("   -> 未发现恒定列。")

    print(f"   清洗后维度: {df.shape}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 结果已保存至: {OUTPUT_FILE}")
    print("✅ 第一步完成。")


if __name__ == "__main__":
    main()