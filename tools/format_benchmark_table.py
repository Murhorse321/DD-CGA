# tools/format_benchmark_table.py
"""
把 benchmark_inference.py 产出的 CSV（或任意同列CSV）
转成 Markdown/LaTeX 表格文件，方便直接贴到论文/报告。

默认输入列名：
- batch
- latency_ms
- throughput_sps

用法：
  python tools/format_benchmark_table.py --csv results/bench/inference_benchmark.csv
可选：
  --outdir results/bench/tables --caption "Inference benchmark" --label tab:bench --floatfmt 3 --md_only
"""
# 方式 A：默认（生成 Markdown + LaTeX）
# python tools/format_benchmark_table.py --csv results/bench/inference_benchmark.csv
# results/bench/tables/inference_benchmark.md
# results/bench/tables/inference_benchmark.tex
# 方式 B：自定义参数
# python tools/format_benchmark_table.py ^
#   --csv results/bench/inference_benchmark.csv ^
#   --outdir results/bench/mytables ^
#   --caption "Inference benchmark on RTX 4060 Laptop" ^
#   --label tab:bench_4060 ^
#   --floatfmt 4
# 方式 C：只要其中一种格式
# # 只要 Markdown
# python tools/format_benchmark_table.py --csv results/bench/inference_benchmark.csv --md_only
#
# # 只要 LaTeX
# python tools/format_benchmark_table.py --csv results/bench/inference_benchmark.csv --latex_only

import os
import argparse
import pandas as pd

def fmt_num(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def to_markdown(df, floatfmt=3):
    df_md = df.copy()
    df_md["latency_ms"] = df_md["latency_ms"].apply(lambda v: fmt_num(v, floatfmt))
    df_md["throughput_sps"] = df_md["throughput_sps"].apply(lambda v: fmt_num(v, floatfmt))
    # 重命名列标题更友好
    df_md = df_md.rename(columns={
        "batch": "Batch",
        "latency_ms": "Latency (ms)",
        "throughput_sps": "Throughput (samples/s)"
    })
    # 手写 Markdown 表（pandas to_markdown 也可，这里手工更可控）
    header = "| " + " | ".join(df_md.columns) + " |\n"
    sep    = "| " + " | ".join(["---"] * len(df_md.columns)) + " |\n"
    rows = []
    for _, r in df_md.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in df_md.columns) + " |")
    return header + sep + "\n".join(rows) + "\n"

def to_latex(df, caption="", label="", floatfmt=3):
    df_tex = df.copy()
    df_tex["latency_ms"] = df_tex["latency_ms"].apply(lambda v: fmt_num(v, floatfmt))
    df_tex["throughput_sps"] = df_tex["throughput_sps"].apply(lambda v: fmt_num(v, floatfmt))
    df_tex = df_tex.rename(columns={
        "batch": "Batch",
        "latency_ms": "Latency (ms)",
        "throughput_sps": "Throughput (samples/s)"
    })
    # 生成 LaTeX tabular
    cols = list(df_tex.columns)
    col_spec = "r" * len(cols)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for _, r in df_tex.iterrows():
        row = " & ".join(str(r[c]) for c in cols) + " \\\\"
        lines.append(row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser(description="Format benchmark CSV to Markdown/LaTeX tables.")
    ap.add_argument("--csv", required=True, help="Path to benchmark CSV (columns: batch, latency_ms, throughput_sps)")
    ap.add_argument("--outdir", default=None, help="Output dir (default: same dir as CSV + /tables)")
    ap.add_argument("--caption", default="Inference benchmark on CNNBaseline", help="LaTeX caption")
    ap.add_argument("--label", default="tab:inference_benchmark", help="LaTeX label")
    ap.add_argument("--floatfmt", type=int, default=3, help="Decimal places for numbers")
    ap.add_argument("--md_only", action="store_true", help="Only emit Markdown")
    ap.add_argument("--latex_only", action="store_true", help="Only emit LaTeX")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"batch", "latency_ms", "throughput_sps"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"CSV 缺少列：{needed - set(df.columns)}")

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.normpath(args.csv)), "tables")
    os.makedirs(outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.csv))[0]
    md_path = os.path.join(outdir, f"{base}.md")
    tex_path = os.path.join(outdir, f"{base}.tex")

    if not args.latex_only:
        md = to_markdown(df, floatfmt=args.floatfmt)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"📝 Markdown saved: {md_path}")

    if not args.md_only:
        tex = to_latex(df, caption=args.caption, label=args.label, floatfmt=args.floatfmt)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"📝 LaTeX saved:    {tex_path}")

if __name__ == "__main__":
    main()
