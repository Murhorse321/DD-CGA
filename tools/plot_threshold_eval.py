import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve, roc_curve,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_arrays(y_true_path, y_prob_path):
    y_true = np.load(y_true_path)
    y_prob = np.load(y_prob_path)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"Length mismatch: y_true={y_true.shape[0]}, y_prob={y_prob.shape[0]}")
    if y_prob.ndim != 1:
        raise ValueError("y_prob must be a 1D array of P(class=1).")
    return y_true.astype(int), y_prob.astype(float)

def eval_at_threshold(y_true, y_prob, th):
    y_pred = (y_prob >= th).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"threshold": th, "precision": p, "recall": r, "f1": f, "cm": cm}

def save_confusion_matrix(cm, title, out_png):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot metrics from saved y_true/y_prob with arbitrary thresholds.")
    parser.add_argument("--y_true", required=True, help="Path to y_true.npy")
    parser.add_argument("--y_prob", required=True, help="Path to y_prob.npy (probabilities of class=1)")
    parser.add_argument("--thresholds", type=str, default="0.5",
                        help="Comma-separated thresholds, e.g. '0.5,0.45,0.40'")
    parser.add_argument("--target_recall", type=float, default=None,
                        help="If set (e.g. 0.99), find threshold with recall>=target and best F1.")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds in [0.01,0.99] and save a CSV & curve.")
    parser.add_argument("--outdir", type=str, default=None, help="Output folder; default results/analysis/<timestamp>")
    args = parser.parse_args()

    # output dir
    if args.outdir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        outdir = ensure_dir(os.path.join("results", "analysis", ts))
    else:
        outdir = ensure_dir(args.outdir)

    # load arrays
    y_true, y_prob = load_arrays(args.y_true, args.y_prob)

    # global curves/metrics
    auc = roc_auc_score(y_true, y_prob)
    prec, rec, pr_th = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fpr, tpr, roc_th = roc_curve(y_true, y_prob)

    # save PR curve
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AP={ap:.4f})")
    plt.grid(alpha=0.4, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=150)
    plt.close()

    # save ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], ls="--", c="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.4, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
    plt.close()

    # evaluate at user thresholds
    th_list = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    rows = []
    for th in th_list:
        res = eval_at_threshold(y_true, y_prob, th)
        rows.append(res)
        # report & CM
        y_pred = (y_prob >= th).astype(int)
        cls_report = classification_report(y_true, y_pred, digits=4)
        print(f"\n=== Threshold={th:.3f} ===")
        print(cls_report)
        save_confusion_matrix(res["cm"],
                              title=f"Confusion Matrix (th={th:.2f})",
                              out_png=os.path.join(outdir, f"cm_th{th:.2f}.png"))

    # threshold sweep (optional)
    if args.sweep:
        sweep_th = np.linspace(0.01, 0.99, 99)
        recs = []
        for th in sweep_th:
            cur = eval_at_threshold(y_true, y_prob, th)
            recs.append([th, cur["precision"], cur["recall"], cur["f1"]])
        df_sweep = pd.DataFrame(recs, columns=["threshold", "precision", "recall", "f1"])
        df_sweep.to_csv(os.path.join(outdir, "threshold_sweep.csv"), index=False)

        # plot F1 vs threshold
        plt.figure(figsize=(6, 4))
        plt.plot(df_sweep["threshold"], df_sweep["f1"], label="F1")
        plt.plot(df_sweep["threshold"], df_sweep["precision"], label="Precision", alpha=0.7)
        plt.plot(df_sweep["threshold"], df_sweep["recall"], label="Recall", alpha=0.7)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Sweep")
        plt.grid(alpha=0.4, ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "threshold_sweep.png"), dpi=150)
        plt.close()

        # auto best F1
        best_idx = df_sweep["f1"].idxmax()
        best_row = df_sweep.iloc[best_idx]
        print(f"\n🔧 Best F1 in sweep: th={best_row['threshold']:.3f} | "
              f"F1={best_row['f1']:.4f} | P={best_row['precision']:.4f} | R={best_row['recall']:.4f}")

        # target recall
        if args.target_recall is not None:
            df_cand = df_sweep[df_sweep["recall"] >= float(args.target_recall)]
            if len(df_cand):
                # among those, choose best F1
                ridx = df_cand["f1"].idxmax()
                row = df_cand.loc[ridx]
                print(f"🎯 Among recall≥{args.target_recall:.3f}: th={row['threshold']:.3f} "
                      f"| F1={row['f1']:.4f} | P={row['precision']:.4f} | R={row['recall']:.4f}")
            else:
                print(f"⚠️ No threshold achieves recall≥{args.target_recall:.3f} in sweep range.")

    # save summary csv for user thresholds
    if rows:
        out_csv = os.path.join(outdir, "selected_thresholds_metrics.csv")
        flat = []
        for r in rows:
            tn, fp, fn, tp = r["cm"][0,0], r["cm"][0,1], r["cm"][1,0], r["cm"][1,1]
            flat.append({
                "threshold": r["threshold"],
                "precision": r["precision"],
                "recall":    r["recall"],
                "f1":        r["f1"],
                "tn": tn, "fp": fp, "fn": fn, "tp": tp
            })
        pd.DataFrame(flat).to_csv(out_csv, index=False)
        print(f"\n📝 Saved metrics for selected thresholds -> {out_csv}")

    print(f"\n✅ All figures saved to: {outdir}")

if __name__ == "__main__":
    main()
