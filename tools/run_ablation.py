# tools/run_ablation.py
import os
import sys
import json
import shutil
import argparse
from copy import deepcopy
from datetime import datetime

import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def collect_summary(summary_path: str):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_and_capture_summary(config_path: str) -> str:
    """
    启动训练（training/train_cnn_gru.py --config <config_path>），
    并从 stdout 中**只解析当前这次 run** 的 summary 路径。

    优先匹配：
      1) 'Saved summary to <path>/summary.json'
      2) 'Figures:' 行，取后面的目录并拼接 '/summary.json'

    不再使用“扫描 results/figures 最新文件”的兜底，避免跨组串线。
    """
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    cmd = [sys.executable, "training/train_cnn_gru.py", "--config", config_path]
    print(">>> Running:", " ".join(cmd))
    p = None
    try:
        p = __import__("subprocess").Popen(
            cmd,
            stdout=__import__("subprocess").PIPE,
            stderr=__import__("subprocess").STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start training process: {e}")

    summary_path = None
    figures_hint = None

    # 逐行读取 stdout，尽早拿到路径
    while True:
        line = p.stdout.readline()
        if not line and p.poll() is not None:
            break
        if not line:
            continue

        # 同步打印到控制台
        print(line, end="")

        # 1) 直接抓 "Saved summary to <.../summary.json>"
        if "Saved summary to " in line:
            path = line.strip().split("Saved summary to ", 1)[-1].strip()
            path = path.replace("\\", "/")
            summary_path = path  # 期望已是 /summary.json

        # 2) 备选：抓 "Figures:" 提示
        if "Figures:" in line:
            # 例： "▶ Figures:  results/figures\\20250918-224322"
            after = line.split("Figures:", 1)[-1].strip()
            figures_hint = after.replace("\\", "/")

    ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"Training failed with exit code {ret}")

    # 优先用“Saved summary to …”
    if summary_path and os.path.isfile(summary_path):
        return summary_path

    # 退而求其次：用 Figures 目录推断
    if figures_hint:
        candidate = figures_hint.rstrip("/") + "/summary.json"
        if os.path.isfile(candidate):
            return candidate

    # 全都没有，明确报错
    raise FileNotFoundError(
        "未在本次 stdout 中解析到 summary 路径。"
        "请确认 training/train_cnn_gru.py 会打印 "
        "'Saved summary to .../summary.json' 或 'Figures: <dir>'。"
    )


def write_compare(outputs, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "compare.csv")
    md_path = os.path.join(out_dir, "compare.md")
    tex_path = os.path.join(out_dir, "compare.tex")

    headers = [
        "exp_name", "sequence_order", "pooling", "bidirectional", "gru_hidden", "lr",
        "Acc", "F1_macro", "PR_AUC", "ROC_AUC",
        "Cfg_th", "Cfg_F1", "Best_th", "Best_F1",
        "Figures"
    ]
    lines = [",".join(headers)]

    def get(d, *keys, default=""):
        cur = d
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k, None)
        return cur if cur is not None else default

    # Markdown header
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")

    # LaTeX table header（简洁版）
    tex = []
    tex.append("\\begin{tabular}{l l l c c c c c c c c c c l}")
    tex.append("\\hline")
    tex.append(" & ".join(headers) + " \\\\")
    tex.append("\\hline")

    for out in outputs:
        s = out["summary"]
        cfg = out["config"]

        exp_name = out["name"]
        params = get(cfg, "model", "params", default={})
        seq = params.get("sequence_order", "row")
        pool = params.get("pooling", "mean")
        bi = params.get("bidirectional", True)
        gh = params.get("gru_hidden", 128)
        lr = get(cfg, "training", "learning_rate", default="")
        test = s.get("test", {})
        acc = test.get("acc", "")
        f1m = test.get("f1_macro", "")
        pr  = test.get("pr_auc", "")
        roc = test.get("roc_auc", "")

        cfg_th = s.get("threshold_cfg", "")
        cfg_f1 = get(s, "threshold_cfg_metrics", "f1", default="")
        best_th = s.get("threshold_best", "")
        best_f1 = get(s, "threshold_best_metrics", "f1", default="")

        figdir = get(s, "paths", "figures_dir", default="")

        # 兼容空值的格式化
        def f4(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return ""
        def f3(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return ""

        row = [
            exp_name, str(seq), str(pool), str(bi), str(gh), str(lr),
            f4(acc), f4(f1m), f4(pr), f4(roc),
            f3(cfg_th), f4(cfg_f1), f3(best_th), f4(best_f1), figdir
        ]
        lines.append(",".join(row))
        md.append("| " + " | ".join(row) + " |")
        tex.append(" & ".join(row) + " \\\\")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    tex.append("\\hline")
    tex.append("\\end{tabular}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    print(f"\n✅ 输出：\n- {csv_path}\n- {md_path}\n- {tex_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=str, default="config/cnn_gru.yaml",
                    help="作为模板的 CNN+GRU YAML 配置路径")
    ap.add_argument("--out_root", type=str, default="results/ablation",
                    help="对比结果输出根目录")
    args = ap.parse_args()

    base = load_yaml(args.base_config)

    # ===== 3 组实验配置（可按需调整）=====
    # B: last pooling, Bi, hid=128, lr=5e-4, row
    B = deepcopy(base)
    B.setdefault("model", {}).setdefault("params", {})
    B["model"]["params"]["pooling"] = "last"
    B["model"]["params"]["sequence_order"] = "row"
    B["model"]["params"]["bidirectional"] = True
    B["model"]["params"]["gru_hidden"] = 128
    B.setdefault("training", {})
    B["training"]["learning_rate"] = 5e-4

    # C: last pooling, Uni, hid=128, lr=5e-4, row
    C = deepcopy(base)
    C.setdefault("model", {}).setdefault("params", {})
    C["model"]["params"]["pooling"] = "last"
    C["model"]["params"]["sequence_order"] = "row"
    C["model"]["params"]["bidirectional"] = False
    C["model"]["params"]["gru_hidden"] = 128
    C.setdefault("training", {})
    C["training"]["learning_rate"] = 5e-4

    # D: last pooling, Bi, hid=128, lr=5e-4, Z-order（可改 "hilbert"）
    D = deepcopy(base)
    D.setdefault("model", {}).setdefault("params", {})
    D["model"]["params"]["pooling"] = "last"
    D["model"]["params"]["sequence_order"] = "z"   # 可改 "hilbert"
    D["model"]["params"]["bidirectional"] = True
    D["model"]["params"]["gru_hidden"] = 128
    D.setdefault("training", {})
    D["training"]["learning_rate"] = 5e-4

    variants = [
        ("B_GRU_last_bi_row_lr5e-4", B),
        ("C_GRU_last_uni_row_lr5e-4", C),
        ("D_GRU_last_bi_z_lr5e-4", D),
    ]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = os.path.join("tmp_cfg", f"ablation_{ts}")
    os.makedirs(work_dir, exist_ok=True)

    results_dir = os.path.join(args.out_root, ts)
    os.makedirs(results_dir, exist_ok=True)

    outputs = []
    for name, cfg in variants:
        # === 为每组注入独立的 log / ckpt 根目录（训练脚本会在其后拼接时间戳）===
        cfg.setdefault("training", {})
        cfg["training"]["log_dir"]  = os.path.join("results", "logs", name)
        cfg["training"]["ckpt_dir"] = os.path.join("results", "checkpoints", name)

        # 写入临时 YAML
        tmp_cfg_path = os.path.join(work_dir, f"{name}.yaml")
        dump_yaml(cfg, tmp_cfg_path)

        # 运行训练，并获取“本组”的 summary.json 路径
        summary_path = run_and_capture_summary(tmp_cfg_path)
        summary = collect_summary(summary_path)

        # === 归档到 results/ablation/<ts>/<name>/ ===
        arch_dir = os.path.join(results_dir, name)
        os.makedirs(arch_dir, exist_ok=True)

        # 1) summary.json
        shutil.copy2(summary_path, os.path.join(arch_dir, "summary.json"))

        # 2) pointers.json（指向原始路径）
        ckpt = summary.get("paths", {}).get("ckpt", "")
        cfg_used = summary.get("paths", {}).get("config", "")
        fig_dir = summary.get("paths", {}).get("figures_dir", "")

        with open(os.path.join(arch_dir, "pointers.json"), "w", encoding="utf-8") as f:
            json.dump({
                "summary_path": summary_path,
                "ckpt": ckpt,
                "config": cfg_used,
                "figures_dir": fig_dir
            }, f, ensure_ascii=False, indent=2)

        # 3) 可选但推荐：复制 ckpt 与 used_config，形成“自包含归档”
        if ckpt and os.path.isfile(ckpt):
            shutil.copy2(ckpt, os.path.join(arch_dir, os.path.basename(ckpt)))
        if cfg_used and os.path.isfile(cfg_used):
            shutil.copy2(cfg_used, os.path.join(arch_dir, "used_config.yaml"))

        outputs.append({"name": name, "config": cfg, "summary": summary})

    write_compare(outputs, results_dir)
    print("\n🎉 三组实验已完成，汇总表已生成。")
    print(f"📁 每组已归档至: {results_dir}/<B|C|D>/")


if __name__ == "__main__":
    main()
