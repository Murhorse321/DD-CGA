# tools/gen_main_table.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def generate_tables():
    # ==========================================
    # 1. 在这里填入你的真实数据
    # ==========================================
    # 提示：你可以从 test_preds.csv 或 summary.json 中找到这些数
    data = [
        {
            "Model": "CNN (Baseline)",
            "Accuracy": 0.9850,
            "Precision": 0.9820,
            "Recall": 0.9810,
            "F1-Score": 0.9815,
            # "Inference Time (ms)": 2.15  # 可选，没有可以删掉
        },
        {
            "Model": "CNN + GRU",
            "Accuracy": 0.9910,
            "Precision": 0.9905,
            "Recall": 0.9900,
            "F1-Score": 0.9902,
            # "Inference Time (ms)": 3.40
        },
        # 你的最终模型 (Proposed)
        {
            "Model": "CNN-GRU-Attn (Ours)",
            "Accuracy": 0.9986,  # 示例数据
            "Precision": 0.9989,
            "Recall": 0.9984,
            "F1-Score": 0.9987,
            # "Inference Time (ms)": 3.65
        },
    ]

    df = pd.DataFrame(data)

    # ==========================================
    # 2. 自动加粗最优值 (LaTeX 逻辑)
    # ==========================================
    # 我们创建一个用于 LaTeX 输出的 DataFrame
    df_tex = df.copy()

    # 需要比较大小的列 (排除 Model 列)
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # 格式化函数：保留4位小数，最大值加粗
    for col in numeric_cols:
        max_val = df[col].max()
        df_tex[col] = df[col].apply(
            lambda x: f"\\textbf{{{x:.4f}}}" if x == max_val else f"{x:.4f}"
        )

    # 推理时间通常越短越好，还是越长越好？通常不用加粗，直接保留2位
    # if "Inference Time (ms)" in df.columns:
    #     df_tex["Inference Time (ms)"] = df["Inference Time (ms)"].apply(lambda x: f"{x:.2f}")

    # ==========================================
    # 3. 输出 Markdown (用于预览)
    # ==========================================
    print("\n📋 [Markdown Preview] 复制到 GitHub/笔记:\n")
    # Markdown 不加粗 LaTeX 代码，只显示数值
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # ==========================================
    # 4. 输出 LaTeX (用于论文)
    # ==========================================
    print("\n\n📄 [LaTeX Code] 复制到论文 main.tex:\n")

    latex_str = df_tex.to_latex(
        index=False,
        escape=False,  # 防止 \textbf 被转义
        column_format="l" + "c" * (len(df.columns) - 1),  # 第一列左对齐，其他居中
        caption="Comparison of performance metrics with baseline models.",
        label="tab:main_results",
        position="htbp"
    )

    # 稍微美化一下 LaTeX (使用三线表 booktabs)
    latex_str = latex_str.replace("\\toprule",
                                  "\\toprule\n\\textbf{Model} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Time(ms)} \\\\")

    print(latex_str)


if __name__ == "__main__":
    generate_tables()