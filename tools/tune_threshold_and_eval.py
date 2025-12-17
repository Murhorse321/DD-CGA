# 读取现有的 config/config.yaml 和已训练好的 best.pth
#
# 只在验证集做两阶段（先粗后细）的阈值扫描，得到 best_th_val
#
# 用该阈值在测试集只评估一次（避免评估泄露）
#
# 导出曲线图、CSV、小对比表，以及 summary.json
# training/tune_threshold_and_eval.py
import os
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import metrics

from training.dataset_loader import get_dataloaders
from models.cnn_baseline import CNNBaseline


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(p1)
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def sweep_thresholds(y_true, y_prob, ths):
    """返回DataFrame和最佳点(F1, th, P, R)"""
    rows = []
    best = (-1.0, 0.5, 0.0, 0.0)  # f1, th, p, r
    for th in ths:
        y_hat = (y_prob >= th).astype(int)
        p = metrics.precision_score(y_true, y_hat, zero_division=0)
        r = metrics.recall_score(y_true, y_hat, zero_division=0)
        f1 = metrics.f1_score(y_true, y_hat)
        rows.append((float(th), float(p), float(r), float(f1)))
        if f1 > best[0]:
            best = (float(f1), float(th), float(p), float(r))
    df = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])
    return df, best  # best = (f1, th, p, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint (best.pth)")
    ap.add_argument("--outdir", default=None, help="output dir (default: results/tuning/<ts>)")
    ap.add_argument("--suppress_scheduler_warning", action="store_true", help="quiet LR scheduler warn")
    args = ap.parse_args()

    # 输出目录
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or os.path.join("results", "tuning", ts)
    os.makedirs(outdir, exist_ok=True)

    # 读取配置并构建模型
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    dropout = float(cfg["training"].get("dropout", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")
    print(f"▶ OutDir: {outdir}")

    # DataLoader（保持与你训练时一致）
    train_loader, val_loader, test_loader = get_dataloaders(args.config)

    model = CNNBaseline(num_classes=2, dropout=dropout).to(device)

    # 加载权重（兼容有/无 weights_only）
    try:
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ Loaded checkpoint: {args.ckpt}")

    # 1) 验证集概率 -> 选阈值
    y_true_val, y_prob_val = predict_proba(model, val_loader, device)

    # 粗搜
    coarse_ths = np.linspace(0.05, 0.95, 37)
    coarse_df, (c_f1, c_th, c_p, c_r) = sweep_thresholds(y_true_val, y_prob_val, coarse_ths)
    coarse_csv = os.path.join(outdir, "threshold_sweep_metrics_coarse_val.csv")
    coarse_df.to_csv(coarse_csv, index=False)
    print(f"💾 (Val) 粗搜已保存: {coarse_csv}")
    print(f"🔎 (Val) Coarse best => th={c_th:.3f} | F1={c_f1:.4f} | P={c_p:.4f} | R={c_r:.4f}")

    # 细搜（±0.05，步长~0.001）
    lo = max(0.0, c_th - 0.05)
    hi = min(1.0, c_th + 0.05)
    if hi - lo < 1e-6:
        lo, hi = max(0.0, c_th - 0.02), min(1.0, c_th + 0.02)
    fine_ths = np.round(np.linspace(lo, hi, int((hi - lo) / 0.001) + 1), 3)
    fine_df, (f_f1, best_th_val, f_p, f_r) = sweep_thresholds(y_true_val, y_prob_val, fine_ths)
    fine_csv = os.path.join(outdir, "threshold_sweep_metrics_fine_val.csv")
    fine_df.to_csv(fine_csv, index=False)
    print(f"💾 (Val) 细搜已保存: {fine_csv}")

    # 合并并画图
    all_df = pd.concat([coarse_df, fine_df]).drop_duplicates("threshold").sort_values("threshold").reset_index(drop=True)
    all_csv = os.path.join(outdir, "threshold_sweep_metrics_all_val.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"💾 (Val) 合并已保存: {all_csv}")

    plt.figure(figsize=(5, 4))
    plt.plot(all_df["threshold"], all_df["f1"], label="F1")
    plt.plot(all_df["threshold"], all_df["precision"], label="Precision")
    plt.plot(all_df["threshold"], all_df["recall"], label="Recall")
    plt.axvline(best_th_val, linestyle="--", label=f"best_th_val={best_th_val:.3f}")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Metrics vs Threshold (VAL)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    val_plot = os.path.join(outdir, "metrics_vs_threshold_VAL.png")
    plt.savefig(val_plot, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"🖼️ (Val) 曲线已保存: {val_plot}")
    print(f"✅ (Val) best_th_val={best_th_val:.3f} | F1={f_f1:.4f} | P={f_p:.4f} | R={f_r:.4f}")

    # 2) 测试集用 best_th_val 只评一次
    y_true_test, y_prob_test = predict_proba(model, test_loader, device)
    np.savez(os.path.join(outdir, "test_preds.npz"),
             y_true_test=y_true_test, y_prob_test=y_prob_test)
    # 新增：写 CSV，方便 bootstrap_ci 使用
    test_csv = os.path.join(outdir, "test_preds.csv")
    pd.DataFrame({
        "y_true": y_true_test,
        "y_prob": y_prob_test,
        "y_pred": (y_prob_test >= best_th_val).astype(int)  # 用验证集最佳阈值
    }).to_csv(test_csv, index=False)
    print(f"💾 Test preds 已保存: {test_csv}")

    y_pred_test = (y_prob_test >= best_th_val).astype(int)
    p = metrics.precision_score(y_true_test, y_pred_test)
    r = metrics.recall_score(y_true_test, y_pred_test)
    f1 = metrics.f1_score(y_true_test, y_pred_test)
    print(f"🎯 (Test@Val-th) th={best_th_val:.3f} => P={p:.4f} R={r:.4f} F1={f1:.4f}")

    # 混淆矩阵
    cm = metrics.confusion_matrix(y_true_test, y_pred_test)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix (Test, th={best_th_val:.3f})")
    cm_path = os.path.join(outdir, f"confusion_matrix_test_th{best_th_val:.3f}.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"🖼️ (Test) 混淆矩阵已保存: {cm_path}")

    # 3) 小对比表：0.50 / 0.45 / best_th_val（在测试集）
    def eval_at(th):
        y_ = (y_prob_test >= th).astype(int)
        return dict(threshold=float(th),
                    precision=float(metrics.precision_score(y_true_test, y_, zero_division=0)),
                    recall=float(metrics.recall_score(y_true_test, y_, zero_division=0)),
                    f1=float(metrics.f1_score(y_true_test, y_)))
    compare_df = pd.DataFrame([
        eval_at(0.50),
        eval_at(0.45),
        eval_at(best_th_val)
    ])
    cmp_csv = os.path.join(outdir, "threshold_compare_test.csv")
    compare_df.to_csv(cmp_csv, index=False)
    print(f"💾 阈值对比表已保存: {cmp_csv}")

    # 4) 总结信息
    summary = {
        "best_th_val": best_th_val,
        "val_best": {"f1": f_f1, "precision": f_p, "recall": f_r},
        "test_at_best_val": {"f1": f1, "precision": p, "recall": r},
        "ckpt": args.ckpt,
        "config": args.config,
        "outdir": outdir,
        "timestamp": ts
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"🧾 摘要已保存: {os.path.join(outdir, 'summary.json')}")


if __name__ == "__main__":
    main()
