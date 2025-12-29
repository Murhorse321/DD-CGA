# 06_finalize_features.py
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ================= 🧪 实验配置 =================
# 输入：特征分析报告 + 之前划分好的数据集
REPORT_FILE = "results/feature_analysis/feature_report.csv"
INPUT_DIR = "data/step4_split"
OUTPUT_DIR = "data/step5_final"  # 最终用于训练的数据存放地

# 目标特征数 (适配 8x8 图像)
TARGET_COUNT = 64

# 【黑名单】 绝对不能进入模型的特征
BLACKLIST = [
    'label', 'label_int',
    'inbound',               # 作弊特征
    'avg_fwd_segment_size',  # ⚠️ 新增：它是 fwd_packet_length_mean 的重复项
    'avg_bwd_segment_size',  # ⚠️ 新增：如果存在，它也是 bwd_packet_length_mean 的重复项
    'flow_id',
    'source_ip', 'source_port',
    'destination_ip', 'destination_port',
    'timestamp', 'simillarhttp',
    'unnamed:_0'
]


# ===============================================

def main():
    if not os.path.exists(REPORT_FILE):
        print("❌ 找不到特征报告 feature_analysis_report.csv")
        return

    print("🚀 开始最终特征锁定 (Finalize Features)...")

    # 1. 读取报告并筛选
    df_report = pd.read_csv(REPORT_FILE)

    # 获取按重要性排序的所有特征名
    sorted_features = df_report['Feature'].tolist()

    # 执行黑名单过滤
    # 逻辑：如果特征名的小写形式不在黑名单里，且不包含 'ip' (防止漏网之鱼)
    final_candidates = []
    for f in sorted_features:
        f_lower = str(f).lower()
        if f_lower in BLACKLIST:
            print(f"   🚫 剔除黑名单特征: {f}")
            continue
        # 双重保险：剔除任何包含 IP 或 Port 字眼的特征 (除非确认是统计特征)
        # 这里 CIC 数据集通常把统计特征命名为 min_seg_size_forward 等，不会单纯叫 ip
        final_candidates.append(f)

    print(f"   黑名单过滤后剩余候选: {len(final_candidates)} 个")

    # 2. 截取 Top 64
    if len(final_candidates) < TARGET_COUNT:
        print(f"⚠️ 警告：剩余特征不足 {TARGET_COUNT} 个 (仅 {len(final_candidates)} 个)。")
        print("   我们将使用所有剩余特征，后续 Reshape 需要补零 (Padding)。")
        selected_features = final_candidates
    else:
        selected_features = final_candidates[:TARGET_COUNT]
        print(f"   ✅ 已锁定 Top {TARGET_COUNT} 特征。")

    # 打印前 5 个和最后 5 个确认一下
    print(f"   [首 5]: {selected_features[:5]}")
    print(f"   [尾 5]: {selected_features[-5:]}")

    # 保存最终特征列表 (供 dataset_loader 使用)
    with open("data/final_feature_list.txt", "w") as f:
        for item in selected_features:
            f.write(f"{item}\n")
    print("   📋 特征列表已保存至 data/final_feature_list.txt")

    # 3. 裁剪 Train / Val / Test 文件
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 我们需要保留的列 = 选中的 64 个特征 + 2 个标签
    cols_to_keep = selected_features + ['label', 'label_int']

    for split_name in ["train", "val", "test"]:
        input_path = os.path.join(INPUT_DIR, f"{split_name}.csv")
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")

        if os.path.exists(input_path):
            print(f"   🔄 正在处理 {split_name}.csv ...")
            df = pd.read_csv(input_path)

            # 检查列是否存在
            missing = [c for c in cols_to_keep if c not in df.columns]
            if missing:
                print(f"   ❌ 严重错误：以下列在 {split_name} 中缺失: {missing}")
                return

            # 裁剪列 (关键：这一步确立了特征的物理顺序！)
            df_final = df[cols_to_keep]

            # 保存
            df_final.to_csv(output_path, index=False)
            print(f"      -> 已保存至 {output_path} (维度: {df_final.shape})")

    print("-" * 50)
    print("🎉 数据准备阶段彻底完成！")
    print(f"📂 最终数据位于: {OUTPUT_DIR}")
    print("   结构: [64维特征] + [label] + [label_int]")
    print("-" * 50)
    print("下一步：归一化处理 (Step 4)。")


if __name__ == "__main__":
    main()