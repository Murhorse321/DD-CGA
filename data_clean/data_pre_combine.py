import os
import zipfile
import pandas as pd
from tqdm import tqdm
#将cicddos2019数据集中的csv文件合并成一个文件
# 设置路径
base_dir = r"D:\Desktop\CIC-DDoS\CSVs"
extract_dir = os.path.join(base_dir, "extracted")
os.makedirs(extract_dir, exist_ok=True)

# 解压所有zip文件
for zip_file in os.listdir(base_dir):
    if zip_file.endswith(".zip"):
        zip_path = os.path.join(base_dir, zip_file)
        target_dir = os.path.join(extract_dir, os.path.splitext(zip_file)[0])
        os.makedirs(target_dir, exist_ok=True)
        print(f"正在解压: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

# 收集所有CSV路径
csv_files = []
for root, _, files in os.walk(extract_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print(f"找到 {len(csv_files)} 个csv文件，开始合并...")

output_path = os.path.join(base_dir, "CICDDoS2019_Merged.csv")
first_file = True

for csv_file in tqdm(csv_files):
    try:
        # ✅ 删除 low_memory 参数
        chunk_iter = pd.read_csv(
            csv_file,
            engine="python",
            dtype=str,
            chunksize=100_000
        )

        for chunk in chunk_iter:
            chunk.to_csv(output_path, mode='a', header=first_file, index=False)
            first_file = False

    except Exception as e:
        print(f"❌ 读取失败: {csv_file} 错误: {e}")


print("✅ 合并完成。结果保存为:", output_path)
