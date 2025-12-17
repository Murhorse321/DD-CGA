#聚合脚本收集多次实验的 F1/Precision/Recall 均值±标准差
#读取summary.json文件将所有的结果汇总起来
# tools/aggregate_summaries.py
#如果不给参数就统计所有 summary.json，给了参数就只统计指定的
# tools/aggregate_summaries.py
#运行配置
# python tools/aggregate_summaries.py ^
#   results/tuning/20250908-142942/summary.json ^
#   -p "results/tuning/20250909*/summary.json" ^
#   -o results/tuning/my_agg_0909.csv

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

def stat_str(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "NA"
    return f"{a.mean():.4f} ± {a.std(ddof=0):.4f}"

def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    # 兼容/校验关键字段
    req_root = ["outdir", "best_th_val", "test_at_best_val"]
    for k in req_root:
        if k not in s:
            raise ValueError(f"{path} 缺少关键字段: {k}")
    req_metrics = ["f1", "precision", "recall"]
    for k in req_metrics:
        if k not in s["test_at_best_val"]:
            raise ValueError(f"{path} 缺少 test_at_best_val.{k}")
    row = {
        "summary_path": os.path.normpath(path),
        "outdir": s["outdir"],
        "best_th_val": float(s["best_th_val"]),
        "test_f1": float(s["test_at_best_val"]["f1"]),
        "test_p": float(s["test_at_best_val"]["precision"]),
        "test_r": float(s["test_at_best_val"]["recall"]),
    }
    return row

def gather_files(positional_files, patterns):
    files = []
    # 显式文件
    if positional_files:
        files.extend(positional_files)
    # 通配符 pattern
    for pat in patterns or []:
        files.extend(glob.glob(pat))
    # 默认：全量扫描
    if not files:
        files = glob.glob(os.path.join("results", "tuning", "*", "summary.json"))
    # 去重&排序
    files = sorted(set(os.path.normpath(p) for p in files))
    return files

def main():
    ap = argparse.ArgumentParser(
        description="聚合 tune_threshold_and_eval.py 产生的 summary.json，计算均值±标准差等统计。"
    )
    ap.add_argument(
        "files", nargs="*", help="想要统计的 summary.json 文件路径（可多个）。不填则按默认目录自动扫描。"
    )
    ap.add_argument(
        "--pattern", "-p", action="append",
        help="可选：通配符模式（可多次提供），如 -p 'results/tuning/20250908*/summary.json'"
    )
    ap.add_argument(
        "--out", "-o", default=os.path.join("results", "tuning", "aggregate_summary.csv"),
        help="输出CSV路径（默认：results/tuning/aggregate_summary.csv）"
    )
    args = ap.parse_args()

    files = gather_files(args.files, args.pattern)
    if not files:
        print("❌ 没有找到任何 summary.json。请检查：\n"
              "  1) 是否已运行 tools/tune_threshold_and_eval.py 生成结果；\n"
              "  2) 传入的路径/模式是否正确。", file=sys.stderr)
        sys.exit(1)

    rows = []
    bad = []
    for p in files:
        try:
            rows.append(load_summary(p))
        except Exception as e:
            bad.append((p, str(e)))

    if bad:
        print("⚠️ 以下文件解析失败（已跳过）：")
        for p, msg in bad:
            print(f"  - {p}: {msg}")

    if not rows:
        print("❌ 没有可用的 summary 记录。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("outdir").reset_index(drop=True)

    # 打印逐次实验
    print("\n=== 每次实验（按 outdir 排序）===")
    print(df.to_string(index=False))

    # 统计
    f1 = df["test_f1"].to_numpy()
    p  = df["test_p"].to_numpy()
    r  = df["test_r"].to_numpy()
    n  = len(df)

    print("\n=== 汇总统计 ===")
    print(f"样本数 n = {n}")
    print(f"F1 (均值±标准差): {stat_str(f1)} | min={f1.min():.4f} max={f1.max():.4f}")
    print(f"P  (均值±标准差): {stat_str(p)}  | min={p.min():.4f}  max={p.max():.4f}")
    print(f"R  (均值±标准差): {stat_str(r)}  | min={r.min():.4f}  max={r.max():.4f}")

    # 保存CSV
    out_path = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n💾 已保存: {out_path}")

if __name__ == "__main__":
    main()
