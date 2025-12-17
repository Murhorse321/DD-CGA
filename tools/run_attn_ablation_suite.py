# tools/run_attn_ablation_suite.py
# -*- coding: utf-8 -*-
"""
一键运行三份 Attention 拨杆配置（ab1/ab2/ab3），并汇总结果。
- 依次调用 training/train_cnn_gru_att.py --config <yaml>
- 根据 YAML 中 training.run_root 找到最新的 ATT_XXXX 目录
- 读取每次的 summary.json / test_preds.csv / test_probs.csv
- 生成 results/ablation_attn/attn_ablation_summary.(csv|md|tex)
- （可选）--baseline_preds 与每次 run 做 CI（F1+PR-AUC），输出到 results/ablation_attn/ci/

用法示例：
python tools/run_attn_ablation_suite.py \
  --configs config/cnn_gru_att_ab1.yaml config/cnn_gru_att_ab2.yaml config/cnn_gru_att_ab3.yaml \
  --python_exe "E:/anaconda3/envs/CNN-GRU-Attention/python.exe" \
  --train_script training/train_cnn_gru_att.py \
  --baseline_preds results/tuning/20250924-172520/test_preds.csv
"""
import argparse, os, sys, subprocess, json, time, glob, yaml
from pathlib import Path
import pandas as pd

def run_cmd(cmd:list, cwd=None):
    print(">>>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"命令执行失败（returncode={p.returncode}）：\n{cmd}\n{p.stdout}")
    return p.stdout

def load_yaml(path:str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def newest_run_dir(run_root:Path):
    if not run_root.exists():
        return None
    cands = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("ATT_")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def read_summary(run_dir:Path):
    sj = run_dir / "summary.json"
    if not sj.exists():
        return None
    with open(sj, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True); return p

def ci_compare(python_exe, ci_script, preds_a, preds_b, metric, out_path):
    out_dir = Path(out_path).parent
    ensure_dir(out_dir)
    cmd = [
        python_exe, ci_script,
        "--preds_a", preds_a,
        "--preds_b", preds_b,
        "--metric", metric,
        "--paired", "true",
        "--n_boot", "10000",
        "--out", out_path
    ]
    run_cmd(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="三个 YAML 配置文件路径，推荐按 ab1/ab2/ab3 顺序传入")
    ap.add_argument("--python_exe", default="python", help="Python 可执行文件路径")
    ap.add_argument("--train_script", default="training/train_cnn_gru_att.py", help="训练脚本路径")
    ap.add_argument("--workdir", default=".", help="项目根目录（相对/绝对均可）")
    ap.add_argument("--ci_script", default="tools/bootstrap_ci_gru.py", help="CI 脚本路径（可选）")
    ap.add_argument("--baseline_preds", default="", help="（可选）CNN 基线的 test_preds.csv，用于与每次 run 做 CI")
    ap.add_argument("--export_tex", action="store_true", help="同时导出 LaTeX 表格")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    out_root = ensure_dir(workdir / "results" / "ablation_attn")
    rows = []
    run_infos = []

    for cfg_path in args.configs:
        cfg_path = Path(cfg_path).resolve()
        cfg = load_yaml(str(cfg_path))
        run_root = Path(cfg["training"].get("run_root", "results/tuning_gru_attn"))
        # 1) 运行训练
        cmd = [args.python_exe, args.train_script, "--config", str(cfg_path)]
        run_cmd(cmd, cwd=str(workdir))
        # 2) 找 run_dir（训练结束后最新的 ATT_ 目录）
        rd = newest_run_dir(workdir / run_root)
        if rd is None:
            raise FileNotFoundError(f"未在 {run_root} 下找到 ATT_* 目录。请检查训练是否成功。")
        print(f"[OK] 本次 run_dir = {rd}")
        # 3) 读取 summary.json
        sm = read_summary(rd)
        if sm is None:
            raise FileNotFoundError(f"{rd}/summary.json 不存在。")
        # 4) 记录关键指标
        test = sm.get("test_at_best", {})
        val_best = sm.get("val_f1_best", None)
        rows.append(dict(
            config=os.path.basename(str(cfg_path)),
            run_root=str(run_root),
            run_dir=str(rd),
            val_f1_best=val_best,
            test_f1=test.get("f1", None),
            test_precision=test.get("precision", None),
            test_recall=test.get("recall", None),
            test_roc_auc=test.get("roc_auc", None),
            test_pr_auc=test.get("pr_auc", None),
            threshold=test.get("threshold", None),
        ))
        run_infos.append(dict(cfg=str(cfg_path), run_dir=str(rd)))

    # === 汇总表 ===
    df = pd.DataFrame(rows)
    csv_path = out_root / "attn_ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    # Markdown
    md_path = out_root / "attn_ablation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| config | val_f1_best | test_f1 | precision | recall | roc_auc | pr_auc | th* | run_dir |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write("| {config} | {val_f1_best:.4f} | {test_f1:.4f} | {test_precision:.4f} | {test_recall:.4f} | {test_roc_auc:.4f} | {test_pr_auc:.4f} | {threshold:.3f} | {run_dir} |\n".format(
                config=r["config"],
                val_f1_best=r["val_f1_best"],
                test_f1=r["test_f1"],
                test_precision=r["test_precision"],
                test_recall=r["test_recall"],
                test_roc_auc=r["test_roc_auc"],
                test_pr_auc=r["test_pr_auc"],
                threshold=r["threshold"],
                run_dir=r["run_dir"],
            ))
    print(f"✅ 汇总完成：\n- {csv_path}\n- {md_path}")

    # LaTeX（可选）
    if args.export_tex:
        tex_path = out_root / "attn_ablation_summary.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{l r r r r r r r l}\\toprule\n")
            f.write("config & val F1 & test F1 & P & R & ROC-AUC & PR-AUC & $\\mathrm{th}^*$ & run dir \\\\\n\\midrule\n")
            for r in rows:
                f.write(f"{r['config']} & {r['val_f1_best']:.4f} & {r['test_f1']:.4f} & {r['test_precision']:.4f} & {r['test_recall']:.4f} & {r['test_roc_auc']:.4f} & {r['test_pr_auc']:.4f} & {r['threshold']:.3f} & {r['run_dir']} \\\\\n")
            f.write("\\bottomrule\\end{tabular}\n")
        print(f"- {tex_path}")

    # === 可选：与基线 CNN 做 CI ===
    if args.baseline_preds:
        ci_dir = ensure_dir(out_root / "ci")
        for info in run_infos:
            rd = Path(info["run_dir"])
            test_preds_b = rd / "test_preds.csv"
            if not test_preds_b.exists():
                print(f"⚠️ 跳过 CI：{test_preds_b} 不存在"); continue
            tag = Path(info["cfg"]).stem + "_" + rd.name
            # F1
            out_f1 = ci_dir / f"cnn_vs_{tag}_f1.json"
            ci_compare(args.python_exe, args.ci_script, args.baseline_preds, str(test_preds_b), "f1", str(out_f1))
            # PR-AUC
            out_pr = ci_dir / f"cnn_vs_{tag}_prauc.json"
            ci_compare(args.python_exe, args.ci_script, args.baseline_preds, str(test_preds_b), "pr_auc", str(out_pr))
        print(f"✅ CI 结果已输出到：{ci_dir}")

if __name__ == "__main__":
    main()
