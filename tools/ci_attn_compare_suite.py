# tools/ci_attn_compare_suite.py
# -*- coding: utf-8 -*-
"""
一键跑 4 条 CI（bootstrap_ci_gru.py）并汇总表格：
1) baseline vs temp12 - f1
2) baseline vs temp12 - pr_auc
3) pos09    vs temp12 - f1
4) pos09    vs temp12 - pr_auc

输出：
- <out_dir>/ci_suite_summary.csv
- <out_dir>/ci_suite_summary.md
- <out_dir>/ci_suite_summary.tex
- 同时保存每条 CI 的 JSON 到 <out_dir>/json/

使用示例（命令行/Windows）：
E:\anaconda3\envs\CNN-GRU-Attention\python.exe tools/ci_attn_compare_suite.py ^
  --baseline_preds results/tuning/20250924-172520/test_preds.csv ^
  --pos09_preds    results/tuning_gru_attn_pos09/ATT_20251015-092351/test_preds.csv ^
  --temp12_preds   results/tuning_gru_attn_temp12/ATT_20251015-094942/test_preds.csv ^
  --out_dir        results/ci/attn_suite

在 PyCharm 的 “Parameters” 可直接粘贴（见本文底部）。
"""
import argparse, json, subprocess, os
from pathlib import Path
import pandas as pd

def run_cmd(cmd:list):
    print(">>>", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="ignore")
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败（returncode={p.returncode}）：\n{cmd}\n{p.stdout}")

def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True); return p

def run_ci(python_exe, ci_script, preds_a, preds_b, metric, out_json):
    cmd = [python_exe, ci_script,
           "--preds_a", preds_a,
           "--preds_b", preds_b,
           "--metric", metric,
           "--paired", "true",
           "--n_boot", "10000",
           "--out", out_json]
    run_cmd(cmd)
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_preds", required=True, help="CNN 基线 test_preds.csv")
    ap.add_argument("--pos09_preds",    required=True, help="Attention(pos_weight=0.9) test_preds.csv")
    ap.add_argument("--temp12_preds",   required=True, help="Attention(temperature=1.2) test_preds.csv")
    ap.add_argument("--python_exe",     default="python", help="Python 解释器（默认当前环境）")
    ap.add_argument("--ci_script",      default="tools/bootstrap_ci_gru.py", help="CI 脚本路径")
    ap.add_argument("--out_dir",        default="results/ci/attn_suite", help="汇总输出目录")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    json_dir = ensure_dir(out_dir / "json")

    jobs = [
        # (标签, preds_a, preds_b, metric, json_name)
        ("baseline_vs_temp12_f1",    args.baseline_preds, args.temp12_preds, "f1",     "baseline_vs_temp12_f1.json"),
        ("baseline_vs_temp12_prauc", args.baseline_preds, args.temp12_preds, "pr_auc", "baseline_vs_temp12_prauc.json"),
        ("pos09_vs_temp12_f1",       args.pos09_preds,    args.temp12_preds, "f1",     "pos09_vs_temp12_f1.json"),
        ("pos09_vs_temp12_prauc",    args.pos09_preds,    args.temp12_preds, "pr_auc", "pos09_vs_temp12_prauc.json"),
    ]

    rows = []
    for tag, pa, pb, metric, jname in jobs:
        out_json = json_dir / jname
        res = run_ci(args.python_exe, args.ci_script, pa, pb, metric, str(out_json))
        sc = res["scores"]; ci_lo, ci_hi = sc["ci_95"]
        rows.append(dict(
            comparison=tag,
            metric=res["metric"],
            paired=res["paired"],
            n_boot=res["n_boot"],
            A_all=sc["A_all"],
            B_all=sc["B_all"],
            delta=sc["delta"],
            ci_low=ci_lo,
            ci_high=ci_hi,
            significant=sc["significant"],
            preds_a=res["inputs"]["preds_a"],
            preds_b=res["inputs"]["preds_b"],
        ))

    df = pd.DataFrame(rows)
    csv_path = out_dir / "ci_suite_summary.csv"
    df.to_csv(csv_path, index=False)

    # Markdown
    md_path = out_dir / "ci_suite_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| comparison | metric | A_all | B_all | Δ (B−A) | 95% CI | significant |\n")
        f.write("|---|---|---:|---:|---:|---|:---:|\n")
        for r in rows:
            ci = f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
            f.write(f"| {r['comparison']} | {r['metric']} | {r['A_all']:.6f} | {r['B_all']:.6f} | {r['delta']:.6f} | {ci} | {'✅' if r['significant'] else '—'} |\n")

    # LaTeX
    tex_path = out_dir / "ci_suite_summary.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l r r r l c}\\toprule\n")
        f.write("comparison & metric & A\\_all & B\\_all & $\\Delta$ (B$-$A) & 95\\% CI & sig \\\\\n\\midrule\n")
        for r in rows:
            ci = f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
            sig = "\\checkmark" if r["significant"] else "-"
            f.write(f"{r['comparison']} & {r['metric']} & {r['A_all']:.6f} & {r['B_all']:.6f} & {r['delta']:.6f} & {ci} & {sig} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

    print("✅ 已完成：")
    print(f"- {csv_path}")
    print(f"- {md_path}")
    print(f"- {tex_path}")
    print(f"- 以及 JSON 明细：{json_dir}")

if __name__ == "__main__":
    main()
