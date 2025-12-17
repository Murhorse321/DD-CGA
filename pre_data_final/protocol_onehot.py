# import pandas as pd
# import os
#
# # 输入文件路径（请自行修改）
# input_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_balanced_dataset_cleaned.csv'
# output_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\onehot_chunks'
# os.makedirs(output_dir, exist_ok=True)
#
# # 分块大小
# chunk_size = 1000000  # 每次读取 100 万行，可根据内存调整
#
# chunk_iter = pd.read_csv(input_path, chunksize=chunk_size)
# for i, chunk in enumerate(chunk_iter):
#     print(f"🔍 正在处理第 {i + 1} 块数据...")
#
#     # 独热编码 protocol
#     chunk = pd.get_dummies(chunk, columns=['protocol'], prefix='protocol')
#
#     # 保存当前块
#     output_path = os.path.join(output_dir, f'encoded_chunk_{i + 1}.csv')
#     chunk.to_csv(output_path, index=False)
#     print(f"✅ 第 {i + 1} 块已保存到 {output_path}")
#
# print("\n🎉 所有块已完成独热编码并保存！")
#
# import pandas as pd
# import os
#
# # 📁 输入文件夹路径（请修改）
# input_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\onehot_chunks'
# output_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\encoded_chunks_fixed'
# os.makedirs(output_dir, exist_ok=True)
#
# # 遍历所有已编码的 csv 文件
# all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
#
# for i, file in enumerate(all_files):
#     file_path = os.path.join(input_dir, file)
#     print(f"🔍 正在处理文件: {file_path}")
#
#     # 读取数据
#     df = pd.read_csv(file_path)
#
#     # 找出所有 protocol 独热编码列
#     protocol_columns = [col for col in df.columns if col.startswith('protocol_')]
#
#     # 布尔值转换为整数（0 / 1）
#     df[protocol_columns] = df[protocol_columns].astype(int)
#
#     # 保存处理后的文件
#     output_path = os.path.join(output_dir, file)
#     df.to_csv(output_path, index=False)
#     print(f"✅ 已保存处理后的文件到: {output_path}")
#
# print("\n🎉 全部文件已处理完毕！")
#
#



import pandas as pd
import os

# 小块文件夹路径（请自行修改）
input_dir = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\encoded_chunks_fixed'
output_path = r'D:\Desktop\C_G_A\CNN_GRU_ATTENTION\final_balanced_dataset_onehot_Pro.csv'

all_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')])

chunks = []
for file in all_files:
    print(f"📂 正在读取: {file}")
    chunks.append(pd.read_csv(file))

final_df = pd.concat(chunks, ignore_index=True)
final_df.to_csv(output_path, index=False)
print(f"\n✅ 全部小块已合并并保存为: {output_path}")
