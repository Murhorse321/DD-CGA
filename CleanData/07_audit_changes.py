# 07_audit_changes.py
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ================= 🔧 配置区域 =================
# 输入文件
REPORT_FILE = "results/feature_analysis/feature_report.csv"  # 包含所有分析过的特征
FINAL_LIST_FILE = "data/final_feature_list.txt"  # 最终幸存者

# 输出文件
AUDIT_LOG = "removed_features_audit.txt"

# 必须与 06 脚本中的黑名单保持一致，用于判定原因
BLACKLIST = [
    'label', 'label_int',
    'inbound',
    'avg_fwd_segment_size',
    'avg_bwd_segment_size',
    'flow_id',
    'source_ip', 'source_port',
    'destination_ip', 'destination_port',
    'timestamp', 'simillarhttp',
    'unnamed:_0'
]


# ===============================================

def main():
    if not os.path.exists(REPORT_FILE) or not os.path.exists(FINAL_LIST_FILE):
        print("❌ 错误：找不到输入文件。请确保 Step 3.5 (锁定特征) 已完成。")
        return

    print("🚀 开始特征变动审计 (Audit)...")

    # 1. 读取数据
    df_report = pd.read_csv(REPORT_FILE)
    all_analyzed_features = set(df_report['Feature'].tolist())

    with open(FINAL_LIST_FILE, 'r') as f:
        final_features = set([line.strip() for line in f if line.strip()])

    print(f"   📊 原始分析特征数: {len(all_analyzed_features)}")
    print(f"   ✅ 最终保留特征数: {len(final_features)}")

    # 2. 计算被剔除的集合
    # 注意：这里计算的是【参与了分析但被剔除】的数值型特征
    # 像 flow_id 这种字符串特征早已被 exclude 掉，可能不在 report 里，我们单独处理
    removed_features = all_analyzed_features - final_features

    print(f"   🗑️ 本轮共剔除特征: {len(removed_features)} 个")

    # 3. 分类原因
    reason_blacklist = []
    reason_low_rank = []

    for feat in removed_features:
        if feat in BLACKLIST or feat.lower() in BLACKLIST:
            reason_blacklist.append(feat)
        else:
            reason_low_rank.append(feat)

    # 4. 生成审计报告
    with open(AUDIT_LOG, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("       特征选择审计日志 (Feature Selection Audit)\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"原始候选特征数 (Numeric): {len(all_analyzed_features)}\n")
        f.write(f"最终保留特征数 (Top 64): {len(final_features)}\n")
        f.write(f"被剔除特征总数: {len(removed_features)}\n\n")

        f.write("-" * 30 + "\n")
        f.write("【类型 A】手动黑名单剔除 (Manual Blacklist)\n")
        f.write("原因：涉及作弊 (Inbound)、重复 (Avg Segment) 或身份信息。\n")
        f.write("-" * 30 + "\n")
        if reason_blacklist:
            for item in sorted(reason_blacklist):
                # 尝试从报告里找排名和分数
                row = df_report[df_report['Feature'] == item]
                if not row.empty:
                    rank = row.iloc[0]['Rank']
                    score = row.iloc[0]['Importance']
                    f.write(f"[Rank {rank:02d}] {item:<30} (Score: {score:.4f})\n")
                else:
                    f.write(f"[Unknown] {item}\n")
        else:
            f.write("(无数值型特征被黑名单剔除)\n")

        f.write("\n" + "-" * 30 + "\n")
        f.write("【类型 B】低重要性自动截断 (Low Importance Cut-off)\n")
        f.write("原因：在随机森林重要性排序中位于 Top 64 之外。\n")
        f.write("-" * 30 + "\n")

        # 按排名排序输出
        low_rank_details = []
        for item in reason_low_rank:
            row = df_report[df_report['Feature'] == item]
            if not row.empty:
                low_rank_details.append((item, row.iloc[0]['Rank'], row.iloc[0]['Importance']))

        # 排序
        low_rank_details.sort(key=lambda x: x[1])  # 按 Rank 排序

        for item, rank, score in low_rank_details:
            f.write(f"[Rank {rank:02d}] {item:<30} (Score: {score:.6f})\n")

    print(f"   💾 审计日志已生成: {AUDIT_LOG}")
    print("   👉 你可以直接复制该文件内容到论文笔记中。")


if __name__ == "__main__":
    main()