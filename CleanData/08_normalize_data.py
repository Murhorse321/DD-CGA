# 08_normalize_data.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import joblib  # 用于保存归一化参数
from sklearn.preprocessing import MinMaxScaler

# ================= 🧪 实验配置 =================
# 输入：Step 3.5 锁定的最终特征数据
INPUT_DIR = "/CleanData/data/step5_final"
# 输出：归一化后的数据 (准备喂给 PyTorch)
OUTPUT_DIR = "results/data/step6_normalized"
# Scaler 保存路径 (重要!)
SCALER_PATH = "results/data/scaler.pkl"

# 归一化范围：[0, 1] 适合转化为灰度图
FEATURE_RANGE = (0, 1)


# ===============================================

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 找不到输入目录 {INPUT_DIR}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("🚀 开始数据归一化 (MinMax Scaling)...")

    # 1. 读取所有数据集
    print("   正在读取 Train / Val / Test ...")
    df_train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(INPUT_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # 2. 区分 特征列 vs 标签列
    # 我们之前已经保证了列结构是 [特征...特征, label, label_int]
    # 自动识别排除列
    exclude_cols = ['label', 'label_int']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    print(f"   检测到特征列数: {len(feature_cols)} (应为 64)")

    # 3. 拟合 Scaler (仅使用训练集!)
    print("   [关键] 正在基于训练集计算 Min/Max ...")
    scaler = MinMaxScaler(feature_range=FEATURE_RANGE)

    # Fit: 这一步计算了每个特征列的 min 和 max
    scaler.fit(df_train[feature_cols])

    # 保存 Scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"   💾 归一化参数已保存至: {SCALER_PATH}")

    # 4. 转换并保存所有数据集
    def process_and_save(df, name):
        # 复制一份，避免修改原变量
        df_scaled = df.copy()

        # 转换特征列
        # 注意：如果有特征值超过了训练集的范围（比如测试集出现了更大的包），
        # MinMax 会把它变成 >1 的数，这是正常的，CNN 能处理。
        # 如果你想强制截断到 1，可以加 clip，但通常不需要。
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])

        # 安全检查：处理极少数可能出现的 NaN (例如某列方差极小计算溢出)
        df_scaled[feature_cols] = df_scaled[feature_cols].fillna(0)

        # 保存
        save_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df_scaled.to_csv(save_path, index=False)
        print(f"   -> {name.upper()} 集已保存: {save_path}")

    process_and_save(df_train, "train")
    process_and_save(df_val, "val")
    process_and_save(df_test, "test")

    print("-" * 50)
    print("🎉 数据预处理全流程完美收官！")
    print(f"📂 最终成品位于: {OUTPUT_DIR}")
    print("   这些 CSV 文件里的数值现在都在 0 到 1 之间。")
    print("   每一行都可以直接 Reshape 成一个 8x8 的灰度图像。")
    print("-" * 50)
    print("下一步计划：")
    print("编写 PyTorch 的 Dataset Loader，把这些 CSV 变成模型能吃的 Tensor。")


if __name__ == "__main__":
    main()