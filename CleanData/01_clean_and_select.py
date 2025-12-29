# 01_clean_and_select.py
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置区域 =================
# 请确保此处路径正确
RAW_DATA_DIR = r"E:\CIC-DDoS\CSVS_chart\CSV-03-11"
OUTPUT_DIR = "data/step1_cleaned_1"

# 批处理大小：如果内存依然报错，可将此数值调小（如 50000）
CHUNK_SIZE = 100000

# 需要严格剔除的特征列表
DROP_COLS = [
    'unnamed:_0',
    'flow_id',
    'source_ip',
    'source_port',
    'destination_ip',
    'destination_port',
    'timestamp',
    'simillarhttp'
]


# ===========================================

def process_chunk(df):
    """
    对单个数据块进行清洗
    """
    # 1. 列名标准化
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 2. 剔除指定列
    existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
    if existing_drop_cols:
        df.drop(columns=existing_drop_cols, inplace=True)

    # 3. 处理 Infinity 和 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 4. 块内去重 (注：全局去重极其耗费内存，在大数据量下通常仅做块内去重或后续处理)
    df.drop_duplicates(inplace=True)

    return df


def clean_csv_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    save_path = os.path.join(output_dir, filename)

    # 如果目标文件已存在，先删除，防止追加写入导致数据重复
    if os.path.exists(save_path):
        os.remove(save_path)

    total_rows = 0
    first_chunk = True

    try:
        # 使用 chunksize 分块读取
        # engine='c' 通常更快，但如果遇到解析错误可尝试 engine='python'
        with pd.read_csv(file_path, encoding='utf-8', chunksize=CHUNK_SIZE, low_memory=False) as reader:
            for chunk in reader:
                # 处理当前块
                cleaned_chunk = process_chunk(chunk)

                if cleaned_chunk.empty:
                    continue

                rows = len(cleaned_chunk)
                total_rows += rows

                # 写入模式：第一块用 'w' 并保留表头，后续块用 'a' 并去除表头
                if first_chunk:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    cleaned_chunk.to_csv(save_path, index=False, mode='w', header=True)
                    first_chunk = False
                else:
                    cleaned_chunk.to_csv(save_path, index=False, mode='a', header=False)

        return filename, total_rows

    except Exception as e:
        print(f"\n❌ 处理文件 {filename} 时依然出错: {e}")
        # 如果出错，建议检查该 CSV 是否损坏
        return filename, 0


def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"错误: 找不到原始数据目录: {RAW_DATA_DIR}")
        return

    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"警告: 在目录 {RAW_DATA_DIR} 下未找到 .csv 文件。")
        return

    print(f"Found {len(csv_files)} csv files, start cleaning (Chunk Mode)...")
    print("-" * 50)

    total_global_rows = 0
    with tqdm(total=len(csv_files)) as pbar:
        for f in csv_files:
            fname, rows = clean_csv_file(f, OUTPUT_DIR)
            total_global_rows += rows
            pbar.set_description(f"Processing {fname}")
            pbar.update(1)

    print("-" * 50)
    print(f"✅ 步骤一 (修正版) 完成！数据已保存至: {OUTPUT_DIR}")
    print(f"📊 总有效样本数: {total_global_rows}")
    print("请检查输出目录的文件大小是否合理。")
    print("确认无报错后，请回复“可以继续”。")


if __name__ == "__main__":
    main()