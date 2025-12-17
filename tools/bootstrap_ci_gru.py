# tools/bootstrap_ci_gru.py
"""
Bootstrap CI (GRU 专用版)
- 比较两组方法（A vs B）在同一测试集上的指标差值的置信区间。
- 推荐使用样本级配对 bootstrap（--paired true），统计力更强。
- 默认读取 test_preds.csv，包含列：y_true, y_pred, y_prob（来自 tune_threshold_and_eval_gru.py）
- 支持 metric: f1 / accuracy / precision / recall / roc_auc / pr_auc
- 对分类指标（f1/acc/prec/rec），优先使用 CSV 里的 y_pred；若提供 --threshold，则用 y_prob >= threshold 重新生成 y_pred。

用法示例：
python tools/bootstrap_ci_gru.py ^
  --preds_a results\\tuned\\CNN_baseline\\test_preds.csv ^
  --preds_b results\\tuning_gru\\C_GRU_last_uni_row_lr5e-4_09241636\\test_preds.csv ^
  --metric f1 --paired true --n_boot 10000 ^
  --out results\\ci\\cnn_vs_gruC.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)


def _load_preds(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Preds file not found: {path}")
    df = pd.read_csv(path)
    # 规范列名（容错：允许大写或空格）
    cols = {c.strip().lower(): c for c in df.columns}
    need_y = "y_true" in cols
    if not need_y:
        raise ValueError(f"{path} 缺少列 y_true")
    # 可选列
    y_true = df[cols["y_true"]].to_numpy()
    y_prob = None
    y_pred = None
    if "y_prob" in cols:
        y_prob = df[cols["y_prob"]].to_numpy()
    if "y_pred" in cols:
        y_pred = df[cols["y_pred"]].to_numpy()
    return pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred})


def _compute_metric(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_prob: Optional[np.ndarray],
    metric: str,
    threshold: Optional[float] = None
) -> float:
    metric = metric.lower()
    if metric in ("roc_auc", "pr_auc"):
        if y_prob is None or np.all(pd.isna(y_prob)):
            raise ValueError(f"metric={metric} 需要 y_prob，但文件中没有 y_prob 列。")
        if metric == "roc_auc":
            return float(roc_auc_score(y_true, y_prob))
        else:
            return float(average_precision_score(y_true, y_prob))
    else:
        # 分类指标：优先使用 y_pred；若阈值提供或 y_pred 缺失，则基于 y_prob 阈值化
        if (y_pred is None or np.all(pd.isna(y_pred))) or (threshold is not None):
            if y_prob is None or np.all(pd.isna(y_prob)):
                raise ValueError(f"metric={metric} 需要 y_pred 或 y_prob+threshold。")
            thr = 0.5 if threshold is None else float(threshold)
            y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            return float(f1_score(y_true, y_pred))
        elif metric in ("acc", "accuracy"):
            return float(accuracy_score(y_true, y_pred))
        elif metric in ("prec", "precision"):
            return float(precision_score(y_true, y_pred, zero_division=0))
        elif metric in ("rec", "recall"):
            return float(recall_score(y_true, y_pred, zero_division=0))
        else:
            raise ValueError(f"未知 metric: {metric}")


def _paired_bootstrap_delta(
    y_true: np.ndarray,
    a_pred: Optional[np.ndarray], a_prob: Optional[np.ndarray],
    b_pred: Optional[np.ndarray], b_prob: Optional[np.ndarray],
    metric: str,
    n_boot: int,
    seed: int,
    threshold: Optional[float]
) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=float)

    # 基线整体值与方法整体值（全样本）
    a_all = _compute_metric(y_true, a_pred, a_prob, metric, threshold)
    b_all = _compute_metric(y_true, b_pred, b_prob, metric, threshold)
    base_delta = b_all - a_all

    idx = np.arange(n)
    for i in range(n_boot):
        sel = rng.choice(idx, size=n, replace=True)
        ya = y_true[sel]
        # 采样对应的预测/概率
        ap = None if a_pred is None else a_pred[sel]
        az = None if a_prob is None else a_prob[sel]
        bp = None if b_pred is None else b_pred[sel]
        bz = None if b_prob is None else b_prob[sel]

        ma = _compute_metric(ya, ap, az, metric, threshold)
        mb = _compute_metric(ya, bp, bz, metric, threshold)
        deltas[i] = mb - ma

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(base_delta), (float(lo), float(hi)), float(a_all), float(b_all)


def _unpaired_bootstrap_delta(
    y_true_a: np.ndarray, a_pred: Optional[np.ndarray], a_prob: Optional[np.ndarray],
    y_true_b: np.ndarray, b_pred: Optional[np.ndarray], b_prob: Optional[np.ndarray],
    metric: str,
    n_boot: int,
    seed: int,
    threshold: Optional[float]
) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    na = len(y_true_a)
    nb = len(y_true_b)
    deltas = np.empty(n_boot, dtype=float)

    a_all = _compute_metric(y_true_a, a_pred, a_prob, metric, threshold)
    b_all = _compute_metric(y_true_b, b_pred, b_prob, metric, threshold)
    base_delta = b_all - a_all

    ia = np.arange(na); ib = np.arange(nb)
    for i in range(n_boot):
        sela = rng.choice(ia, size=na, replace=True)
        selb = rng.choice(ib, size=nb, replace=True)
        ma = _compute_metric(y_true_a[sela],
                             None if a_pred is None else a_pred[sela],
                             None if a_prob is None else a_prob[sela],
                             metric, threshold)
        mb = _compute_metric(y_true_b[selb],
                             None if b_pred is None else b_pred[selb],
                             None if b_prob is None else b_prob[selb],
                             metric, threshold)
        deltas[i] = mb - ma

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(base_delta), (float(lo), float(hi)), float(a_all), float(b_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_a", required=True, help="A 方法的 test_preds.csv 路径（如：CNN 基线）")
    ap.add_argument("--preds_b", required=True, help="B 方法的 test_preds.csv 路径（如：最佳 GRU）")
    ap.add_argument("--metric", default="f1",
                    help="f1 / accuracy / precision / recall / roc_auc / pr_auc（默认 f1）")
    ap.add_argument("--paired", type=str, default="true",
                    help="是否配对自助法（默认 true）。true/false")
    ap.add_argument("--n_boot", type=int, default=10000, help="bootstrap 次数（默认 10000）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument("--threshold", type=float, default=None,
                    help="可选：用于将 y_prob 阈值化为 y_pred（仅作用于 f1/acc/prec/rec）")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    args = ap.parse_args()

    df_a = _load_preds(args.preds_a)
    df_b = _load_preds(args.preds_b)

    # 基本一致性检查（配对时需要同样本量）
    if args.paired.lower() == "true":
        if len(df_a) != len(df_b):
            raise ValueError("配对 bootstrap 需要两侧样本行数一致（同一测试集）。")
        base_delta, ci95, a_all, b_all = _paired_bootstrap_delta(
            y_true=df_a["y_true"].to_numpy(),
            a_pred=None if df_a["y_pred"].isna().all() else df_a["y_pred"].to_numpy(),
            a_prob=None if df_a["y_prob"].isna().all() else df_a["y_prob"].to_numpy(),
            b_pred=None if df_b["y_pred"].isna().all() else df_b["y_pred"].to_numpy(),
            b_prob=None if df_b["y_prob"].isna().all() else df_b["y_prob"].to_numpy(),
            metric=args.metric, n_boot=args.n_boot, seed=args.seed, threshold=args.threshold
        )
    else:
        base_delta, ci95, a_all, b_all = _unpaired_bootstrap_delta(
            y_true_a=df_a["y_true"].to_numpy(),
            a_pred=None if df_a["y_pred"].isna().all() else df_a["y_pred"].to_numpy(),
            a_prob=None if df_a["y_prob"].isna().all() else df_a["y_prob"].to_numpy(),
            y_true_b=df_b["y_true"].to_numpy(),
            b_pred=None if df_b["y_pred"].isna().all() else df_b["y_pred"].to_numpy(),
            b_prob=None if df_b["y_prob"].isna().all() else df_b["y_prob"].to_numpy(),
            metric=args.metric, n_boot=args.n_boot, seed=args.seed, threshold=args.threshold
        )

    out = {
        "metric": args.metric,
        "paired": args.paired.lower() == "true",
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "threshold_used": args.threshold,
        "scores": {
            "A_all": float(a_all),
            "B_all": float(b_all),
            "delta": float(base_delta),
            "ci_95": [float(ci95[0]), float(ci95[1])],
            "significant": not (ci95[0] <= 0.0 <= ci95[1])
        },
        "inputs": {
            "preds_a": args.preds_a,
            "preds_b": args.preds_b
        }
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=== Bootstrap CI (GRU) ===")
    print(f"metric: {args.metric} | paired: {out['paired']} | n_boot: {args.n_boot} | seed: {args.seed}")
    print(f"A_all={a_all:.6f} ; B_all={b_all:.6f} ; Δ={base_delta:.6f} ; 95% CI=({ci95[0]:.6f}, {ci95[1]:.6f})")
    print("significant=", out["scores"]["significant"])
    print(f"→ saved to: {args.out}")


if __name__ == "__main__":
    main()
