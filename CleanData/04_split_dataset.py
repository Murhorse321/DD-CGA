# 04_split_dataset.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= 🧪 实验配置区域 =================
# 输入文件：步骤二生成的平衡数据集
INPUT_FILE = "data/step3_struct_cleaned.csv"
# 输出目录
OUTPUT_DIR = "data/step4_split"

# 划分比例 (Train=80%, Val=10%, Test=10%)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子 (保证论文可复现性的关键)
RANDOM_SEED = 42


# =================================================

def save_split(df, name, output_dir):
    """保存切分后的数据集并打印统计信息"""
    path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(path, index=False)

    # 统计分布
    counts = df['label_int'].value_counts()
    n_benign = counts.get(0, 0)
    n_attack = counts.get(1, 0)
    total = len(df)
    ratio = n_attack / n_benign if n_benign > 0 else 0

    print(f"  -> [{name.upper()}] 集已保存")
    print(f"     路径: {path}")
    print(f"     总数: {total}")
    print(f"     分布: Benign={n_benign}, Attack={n_attack} (Ratio 1:{ratio:.2f})")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        print("请确认你已经成功运行了步骤二的脚本。")
        return

    print(f"🚀 正在读取全量数据: {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # 获取标签用于分层抽样
    # y 包含了每个样本是攻击还是良性
    y = df['label_int']

    print("-" * 50)
    print("✂️ 开始执行分层划分 (Stratified Split)...")

    # 第一刀：切出 训练集 (80%) 和 剩余集 (20%)
    # stratify=y 保证了切分后的两部分中，黑白样本比例与原始数据一致
    train_df, temp_df, y_train, y_temp = train_test_split(
        df, y,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=y
    )

    # 第二刀：将剩余集 (20%) 对半切分为 验证集 (10%) 和 测试集 (10%)
    # 注意：这里的 0.5 是指剩余部分的 50%，即总体的 10%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("-" * 50)
    # 保存并显示统计
    save_split(train_df, "train", OUTPUT_DIR)
    print("-" * 30)
    save_split(val_df, "val", OUTPUT_DIR)
    print("-" * 30)
    save_split(test_df, "test", OUTPUT_DIR)

    print("-" * 50)
    print("✅ 步骤三完成！数据集已严格物理隔离。")
    print("   Train 用于训练，Val 用于早停，Test 用于最终评估。")
    print("   请回复“可以继续”进入最后一步：归一化处理。")


if __name__ == "__main__":
    main()