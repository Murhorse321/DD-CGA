# training/export_misclassified.py
import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="阈值（建议使用tune脚本得到的best_th_val）")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--topn", type=int, default=0, help="每类错分导出前N条（0=导出全部）")
    args = ap.parse_args()

    # 输出目录
    outdir = args.outdir or os.path.join("results", "errors")
    os.makedirs(outdir, exist_ok=True)

    # 读取配置 & 构建模型
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    dropout = float(cfg["training"].get("dropout", 0.5))
    test_csv = cfg["data"]["test_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(args.config)
    model = CNNBaseline(num_classes=2, dropout=dropout).to(device)

    try:
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ Loaded checkpoint: {args.ckpt}")

    # 预测
    y_true, y_prob = predict_proba(model, test_loader, device)
    y_pred = (y_prob >= args.threshold).astype(int)

    # 读取原始测试CSV，拼接预测结果（依赖 DataLoader shuffle=False）
    df = pd.read_csv(test_csv).reset_index(drop=True)
    out = df.copy()
    out["true"] = y_true
    out["prob"] = y_prob
    out["pred"] = y_pred
    out["error_type"] = np.where((out["pred"] == out["true"]), "OK",
                          np.where((out["pred"] == 1) & (out["true"] == 0), "FP", "FN"))

    # 错分集合
    fp = out[out["error_type"] == "FP"].copy()
    fn = out[out["error_type"] == "FN"].copy()

    # 可选：各取前N条（按离阈值的“置信度”排序，越远越“自信”）
    if args.topn > 0:
        fp = fp.assign(margin=(fp["prob"] - args.threshold).abs()).sort_values("margin", ascending=False).head(args.topn)
        fn = fn.assign(margin=(fn["prob"] - args.threshold).abs()).sort_values("margin", ascending=False).head(args.topn)

    # 保存
    base = f"errors_th{args.threshold:.3f}"
    all_path = os.path.join(outdir, f"{base}_all.csv")
    fp_path  = os.path.join(outdir, f"{base}_fp.csv")
    fn_path  = os.path.join(outdir, f"{base}_fn.csv")
    out.to_csv(all_path, index=False)
    fp.to_csv(fp_path, index=False)
    fn.to_csv(fn_path, index=False)

    # 简要统计
    p = metrics.precision_score(y_true, y_pred, zero_division=0)
    r = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred)
    summary = {
        "threshold": args.threshold,
        "precision": p, "recall": r, "f1": f1,
        "n_total": int(len(out)), "n_fp": int(len(fp)), "n_fn": int(len(fn)),
        "paths": {"all": all_path, "fp": fp_path, "fn": fn_path},
        "ckpt": args.ckpt, "config": args.config
    }
    with open(os.path.join(outdir, f"{base}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"📦 导出完成：\n  ALL: {all_path}\n  FP : {fp_path}\n  FN : {fn_path}")
    print(f"🧾 指标：P={p:.4f} R={r:.4f} F1={f1:.4f} | FP={len(fp)} FN={len(fn)} / {len(out)}")


if __name__ == "__main__":
    main()
