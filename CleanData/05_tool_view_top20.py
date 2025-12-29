# view_top20.py
# -*- coding: utf-8 -*-

import os
import pandas as pd


def main():
    # 尝试寻找报告文件 (兼容刚才两个版本的输出路径)
    possible_paths = [
        "results/feature_analysis/feature_report.csv",
        "feature_analysis_report.csv"
    ]

    report_path = None
    for p in possible_paths:
        if os.path.exists(p):
            report_path = p
            break

    if report_path is None:
        print("❌ 错误：未找到 'feature_report.csv'。")
        print("   请确认你是否完整运行了 05_analyze_features.py 脚本。")
        return

    print(f"📖 正在读取报告: {report_path}")
    df = pd.read_csv(report_path)

    # 获取前 20 名
    top20 = df.head(20)

    print("-" * 50)
    print("【Top 20 特征列表】")
    print("-" * 50)

    # 格式化打印
    for index, row in top20.iterrows():
        rank = index + 1
        feat = row['Feature']
        score = row['Importance']
        print(f"{rank:02d}. {feat:<30} (Score: {score:.4f})")

    print("-" * 50)

    # 同时保存到一个 TXT 文件，方便你复制
    out_txt = "top20_features.txt"
    with open(out_txt, "w") as f:
        # 只写入特征名，一行一个
        for feat in top20['Feature']:
            f.write(feat + "\n")

    print(f"✅ Top 20 特征名已保存至: {out_txt}")
    print("   请打开这个 txt 文件，将其内容复制发给我。")


if __name__ == "__main__":
    main()