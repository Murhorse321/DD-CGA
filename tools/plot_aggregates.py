# tools/plot_aggregates.py
"""
将多次实验的 summary.json 或聚合后的 CSV 可视化：
- 箱线图（boxplot）
- 小提琴图（violin）
并可叠加散点抖动，便于观察每次实验的离散分布。

用法示例见文件末尾注释或 README。
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRIC_COLUMNS = {
    "test_f1": "F1",
    "test_p": "Precision",
    "test_r": "Recall",
}

def load_from_summaries(paths):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        rows.append({
            "summary_path": os.path.normpath(p),
            "outdir": s.get("outdir", ""),
            "best_th_val": float(s["best_th_val"]),
            "test_f1": float(s["test_at_best_val"]["f1"]),
            "test_p":  float(s["test_at_best_val"]["precision"]),
            "test_r":  float(s["test_at_best_val"]["recall"]),
        })
    df = pd.DataFrame(rows)
    if "outdir" in df.columns:
        df = df.sort_values("outdir").reset_index(drop=True)
    return df

def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # 兼容常见列名
    rename_map = {}
    for k in ["test_f1", "test_p", "test_r"]:
        if k not in df.columns and k.upper() in df.columns:
            rename_map[k.upper()] = k
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def plot_box_and_violin(df, outdir, title_suffix="All Experiments", jitter=True):
    # 组装绘图数据
    data = [df["test_f1"].values, df["test_p"].values, df["test_r"].values]
    labels = ["F1", "Precision", "Recall"]

    # --- 箱线图 ---
    plt.figure(figsize=(6, 4))
    bp = plt.boxplot(data, labels=labels, showmeans=True)
    # 可选抖动散点
    if jitter:
        rng = np.random.default_rng(42)
        for i, arr in enumerate(data, start=1):
            x = np.random.normal(loc=i, scale=0.03, size=len(arr))
            plt.scatter(x, arr, s=10, alpha=0.6)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Metrics Boxplot ({title_suffix})")
    plt.grid(True, ls="--", alpha=0.4)
    box_path = os.path.join(outdir, "metrics_boxplot.png")
    plt.savefig(box_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"🖼️ 已保存: {box_path}")

    # --- 小提琴图 ---
    plt.figure(figsize=(6, 4))
    vp = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    # x 轴标签
    plt.xticks([1, 2, 3], labels)
    # 可选抖动散点
    if jitter:
        rng = np.random.default_rng(42)
        for i, arr in enumerate(data, start=1):
            x = np.random.normal(loc=i, scale=0.03, size=len(arr))
            plt.scatter(x, arr, s=10, alpha=0.6)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Metrics Violin Plot ({title_suffix})")
    plt.grid(True, ls="--", alpha=0.4)
    violin_path = os.path.join(outdir, "metrics_violin.png")
    plt.savefig(violin_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"🖼️ 已保存: {violin_path}")

def main():
    ap = argparse.ArgumentParser(
        description="将多次实验的 summary.json 或聚合CSV可视化为箱线图/小提琴图"
    )
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--csv", help="聚合后的CSV（如 results/tuning/aggregate_summary.csv）")
    src.add_argument("--files", nargs="+", help="若干个 summary.json 路径")
    ap.add_argument("--pattern", "-p", action="append",
                    help="通配符模式（可多个），如 -p 'results/tuning/20250908*/summary.json'")
    ap.add_argument("--outdir", default=None, help="输出目录，默认基于输入自动推断")
    ap.add_argument("--title", default=None, help="图表标题后缀")
    ap.add_argument("--no-jitter", action="store_true", help="关闭散点抖动")
    args = ap.parse_args()

    # 加载数据
    df = None
    title_suffix = "All Experiments"
    if args.csv:
        df = load_from_csv(args.csv)
        title_suffix = args.title or os.path.basename(os.path.dirname(args.csv)) or "CSV"
        outdir = args.outdir or os.path.join(os.path.dirname(args.csv), "plots")
    else:
        files = []
        if args.files:
            files.extend(args.files)
        if args.pattern:
            for pat in args.pattern:
                files.extend(glob.glob(pat))
        if not files:
            # 默认全扫
            files = glob.glob(os.path.join("results", "tuning", "*", "summary.json"))
        files = sorted(set(os.path.normpath(p) for p in files))
        if not files:
            raise SystemExit("未找到任何 summary.json，请检查路径或先运行 tune_threshold_and_eval.py。")
        df = load_from_summaries(files)
        title_suffix = args.title or f"{len(files)} summaries"
        # 输出目录默认放在第一个 summary 同目录的 plots/
        first_dir = os.path.dirname(files[0])
        outdir = args.outdir or os.path.join(first_dir, "plots")

    # 基本校验
    for col in ["test_f1", "test_p", "test_r"]:
        if col not in df.columns:
            raise SystemExit(f"数据缺少列：{col}")

    ensure_outdir(outdir)
    plot_box_and_violin(df, outdir=outdir, title_suffix=title_suffix, jitter=not args.no_jitter)

if __name__ == "__main__":
    main()
# 使用方法
# 方式 A：直接读取聚合后的 CSV（最简单）
#
# 先用你增强版 aggregate_summaries.py 生成 CSV：
# python tools/aggregate_summaries.py
# # 会得到 results/tuning/aggregate_summary.csv
# 再画图：
# python tools/plot_aggregates.py --csv results/tuning/aggregate_summary.csv
# # 输出到 results/tuning/plots/ 下：
# #   - metrics_boxplot.png
# #   - metrics_violin.png
# 方式 B：指定一组 summary.json（不走 CSV）
# python tools/plot_aggregates.py ^
#   --files results/tuning/20250908-142942/summary.json ^
#           results/tuning/20250908-153300/summary.json ^
#   --outdir results/tuning/plots_selected ^
#   --title "Selected Runs"
