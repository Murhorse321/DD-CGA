import os
import time
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np
import matplotlib.pyplot as plt

from training.dataset_loader import get_dataloaders
from models.cnn_baseline import CNNBaseline


def set_seed(seed: int = 42):
    """固定随机种子，保证可复现"""
    #随机种子设置不同的随机种子保证论文强证据
    #1，21，42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """单个 epoch 的训练循环"""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    评估：返回平均损失、准确率和完整 y_true/y_pred（用于F1等统计）
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += yb.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

    avg_loss = total_loss / total
    acc = total_correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, acc, all_labels, all_preds


@torch.no_grad()
def predict_proba(model, loader, device):
    """
    返回:
        y_true_prob: 真实标签（0/1）
        y_prob: 预测为“攻击(1)”的概率
    """
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(p1)
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main(config_path="config/config.yaml"):
    set_seed(42)

    # 读取配置
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    th_cfg = float(cfg.get("inference", {}).get("threshold", 0.5))

    # 训练超参
    lr       = float(cfg["training"].get("learning_rate", 1e-3))
    epochs   = int(cfg["training"].get("epochs", 20))
    wd       = float(cfg["training"].get("weight_decay", 0.0))
    dropout  = float(cfg["training"].get("dropout", 0.5))
    base_log_dir  = cfg["training"].get("log_dir", "results/logs/cnn_baseline")   # [TS-NEW] 基础路径
    base_ckpt_dir = cfg["training"].get("ckpt_dir", "results/checkpoints")        # [TS-NEW] 基础路径
    patience = int(cfg["training"].get("early_stopping_patience", 5))
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    best_on  = cfg["training"].get("best_on", "val_acc")  # 可为 'val_acc' 或 'val_f1_macro'

    # [TS-NEW] —— 时间戳后缀，避免覆盖旧实验 —— #
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir   = f"{base_log_dir}_{timestamp}"
    ckpt_dir  = os.path.join(base_ckpt_dir, timestamp)
    figures_dir = os.path.join("results/figures", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")
    print(f"▶ LogDir:   {log_dir}")
    print(f"▶ CkptDir:  {ckpt_dir}")
    print(f"▶ Figures:  {figures_dir}")

    # DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(config_path)

    # 模型/损失/优化器
    model = CNNBaseline(num_classes=2, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # 验证集自适应降低学习率（监控 val_acc）
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 早停 & 选择最优指标
    best_metric = -1.0
    best_ckpt = os.path.join(ckpt_dir, "cnn_baseline_best.pth")  # [TS-NEW] 带时间戳目录
    wait = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss, val_acc, y_true_val, y_pred_val = evaluate(model, val_loader, criterion, device)
        val_f1_macro = f1_score(y_true_val, y_pred_val, average="macro")
        val_f1_weighted = f1_score(y_true_val, y_pred_val, average="weighted")

        # 记录
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val",   val_loss,   epoch)
        writer.add_scalar("Acc/Train",  train_acc,  epoch)
        writer.add_scalar("Acc/Val",    val_acc,    epoch)
        writer.add_scalar("F1/Val_macro",    val_f1_macro, epoch)
        writer.add_scalar("F1/Val_weighted", val_f1_weighted, epoch)

        # 调度 LR
        scheduler.step(val_acc)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] "
              f"Train Loss  {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val  Loss   {val_loss:.4f} Acc {val_acc:.4f} F1(macro) {val_f1_macro:.4f} | "
              f"{dt:.1f}s")

        # 保存最优（依据配置 best_on）
        current_metric = val_acc if best_on == "val_acc" else val_f1_macro
        if current_metric > best_metric:
            best_metric = current_metric
            wait = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✅ Best {best_on} improved to {best_metric:.4f}, saved to {best_ckpt}")
        else:
            wait += 1
            print(f"  ⏳ No improvement. wait={wait}/{patience}")
            if wait >= patience:
                print("  🛑 Early stopping triggered.")
                break

    writer.close()

    # 加载最优权重（安全写法，兼容旧版）
    try:
        state_dict = torch.load(best_ckpt, map_location=device, weights_only=True)  # [TS-NEW] 更安全
    except TypeError:
        state_dict = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print(f"\n✅ Loaded best checkpoint: {best_ckpt}")
    # === 基于“验证集”做阈值选择：Precision / FPR 约束下最大化 Recall ===
    # 你可以通过 config 设置约束；没有则使用默认值（示例：P>=0.99）
    MIN_PRECISION = float(cfg.get("inference", {}).get("min_precision", 0.99))  # 例：>=0.99
    MAX_FPR = cfg.get("inference", {}).get("max_fpr", None)  # 例：0.001（千分之一）；None 表示忽略该约束
    if MAX_FPR is not None:
        MAX_FPR = float(MAX_FPR)

    def _fpr_from_confmat(y_true_bin, y_hat_bin):
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_hat_bin).ravel()
        return fp / (fp + tn + 1e-12)

    @torch.no_grad()
    def select_threshold_with_constraints(loader, min_precision=None, max_fpr=None):
        # 返回：best 字典、扫描明细 DataFrame、(y_true, y_prob)
        y_true_bin, y_prob_pos = predict_proba(model, loader, device)
        ths = np.linspace(0.0, 1.0, 1001)  # 0.001 步长，稳定可复现
        best = {"th": 0.5, "recall": -1.0, "precision": 0.0, "fpr": 1.0}
        records = []
        for th in ths:
            y_hat = (y_prob_pos >= th).astype(int)
            p = precision_score(y_true_bin, y_hat, zero_division=0)
            r = recall_score(y_true_bin, y_hat, zero_division=0)
            fpr = _fpr_from_confmat(y_true_bin, y_hat)
            ok = True
            if min_precision is not None and p < min_precision:
                ok = False
            if max_fpr is not None and fpr > max_fpr:
                ok = False
            records.append({
                "threshold": float(th),
                "precision": float(p),
                "recall": float(r),
                "fpr": float(fpr),
                "ok": int(ok)
            })
            # 约束满足时，按 Recall 最大原则选阈值
            if ok and r > best["recall"]:
                best = {"th": float(th), "recall": float(r), "precision": float(p), "fpr": float(fpr)}
        import pandas as pd
        return best, pd.DataFrame(records), (y_true_bin, y_prob_pos)

    best, val_scan_df, (y_true_val_bin, y_prob_val) = select_threshold_with_constraints(
        val_loader, min_precision=MIN_PRECISION, max_fpr=MAX_FPR
    )

    # 保存验证集扫描明细与最佳阈值
    val_scan_csv = os.path.join(figures_dir, "val_threshold_scan_constraints.csv")
    val_scan_df.to_csv(val_scan_csv, index=False)
    import json
    with open(os.path.join(figures_dir, "val_best_threshold.json"), "w") as f:
        json.dump({
            "best_threshold": best["th"],
            "precision": best["precision"],
            "recall": best["recall"],
            "fpr": best["fpr"],
            "min_precision": MIN_PRECISION,
            "max_fpr": MAX_FPR
        }, f, indent=2, ensure_ascii=False)
    print(f"🎯 验证集阈值（约束选择）=> th={best['th']:.3f} | P={best['precision']:.4f} | R={best['recall']:.4f} | FPR={best['fpr']:.6f}")


    # 测试集评估
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    test_f1_macro = f1_score(y_true, y_pred, average="macro")
    test_f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"\n🧪 Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
          f"Test F1(macro): {test_f1_macro:.4f} | Test F1(weighted): {test_f1_weighted:.4f}")
    print("\n📋 Classification Report (Test):")
    print(classification_report(y_true, y_pred, digits=4))

    # 概率用于 AUC/PR-AUC
    y_true_prob, y_prob = predict_proba(model, test_loader, device)
    auc = roc_auc_score(y_true_prob, y_prob)
    prec, rec, _ = precision_recall_curve(y_true_prob, y_prob)
    ap = average_precision_score(y_true_prob, y_prob)
    print(f"🔵 Test ROC-AUC: {auc:.4f}")
    print(f"🟣 Test PR-AUC (Average Precision): {ap:.4f}")

    # 图表保存到带时间戳目录
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix (Test)")
    plt.savefig(os.path.join(figures_dir, "confusion_matrix_test.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)

    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (Test)")
    plt.grid(True, ls='--', alpha=0.4)
    plt.savefig(os.path.join(figures_dir, "pr_curve_test.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"🖼️ 图表已保存到 {figures_dir}")
    # —— 可选：把 best_th 回写到 config.yaml
    cfg.setdefault("inference", {})
    cfg["inference"]["threshold"] = float(best["th"])
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"📝 已写回 inference.threshold={best['th']:.3f} 到 {config_path}")

    # # === 使用配置中的阈值 th_cfg 对测试集再次评估 & 保存一张新的混淆矩阵图 ===
    #
    #
    # y_pred_cfg = (y_prob >= th_cfg).astype(int)
    # p_cfg = precision_score(y_true_prob, y_pred_cfg)
    # r_cfg = recall_score(y_true_prob, y_pred_cfg)
    # f1_cfg = f1_score(y_true_prob, y_pred_cfg)
    # print(f"📌 Using cfg threshold={th_cfg:.3f} => P={p_cfg:.4f} R={r_cfg:.4f} F1={f1_cfg:.4f}")
    #
    # cm_cfg = confusion_matrix(y_true_prob, y_pred_cfg)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_cfg)
    # fig, ax = plt.subplots(figsize=(4, 4))
    # disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    # plt.title(f"Confusion Matrix (Test, th={th_cfg:.2f})")
    # plt.savefig(os.path.join(figures_dir, f"confusion_matrix_test_th{th_cfg:.2f}.png"),
    #             bbox_inches='tight', dpi=150)
    # plt.close(fig)
    # print(f"🖼️ 已保存调阈值后的混淆矩阵：confusion_matrix_test_th{th_cfg:.2f}.png")
    #
    #
    #
    # # ===== 阈值调优（先粗后细）：保存 CSV + 曲线，并打印最优阈值 =====
    #
    # def _sweep_thresholds(y_true_bin, y_prob_pos, ths):
    #     """对给定阈值序列做扫描，返回 records(list[dict]) 与最佳(F1, th, P, R)"""
    #     records = []
    #     best_f1, best_th, best_p, best_r = -1.0, 0.5, 0.0, 0.0
    #     for th in ths:
    #         y_hat = (y_prob_pos >= th).astype(int)
    #         p = precision_score(y_true_bin, y_hat, zero_division=0)
    #         r = recall_score(y_true_bin, y_hat, zero_division=0)
    #         f1 = f1_score(y_true_bin, y_hat)
    #         records.append({"threshold": float(th), "precision": float(p), "recall": float(r), "f1": float(f1)})
    #         if f1 > best_f1:
    #             best_f1, best_th, best_p, best_r = f1, th, p, r
    #     return records, (best_f1, best_th, best_p, best_r)
    #
    # # --- Stage 1: 粗搜（步长≈0.025） ---
    # coarse_ths = np.linspace(0.05, 0.95, 37)  # 0.05, 0.075, ..., 0.95
    # coarse_records, (c_f1, c_th, c_p, c_r) = _sweep_thresholds(y_true_prob, y_prob, coarse_ths)
    #
    # # 保存粗搜 CSV
    # coarse_csv = os.path.join(figures_dir, "threshold_sweep_metrics_coarse.csv")
    # pd.DataFrame(coarse_records).to_csv(coarse_csv, index=False)
    # print(f"💾 粗搜阈值扫描结果已保存: {coarse_csv}")
    # print(f"🔎 Coarse best => th={c_th:.3f} | F1={c_f1:.4f} | P={c_p:.4f} | R={c_r:.4f}")
    #
    # # --- Stage 2: 细搜（在粗搜最优阈值 ±0.05 范围内，步长≈0.001） ---
    # lo = max(0.0, c_th - 0.05)
    # hi = min(1.0, c_th + 0.05)
    # # 避免 lo == hi 的极端情形
    # if hi - lo < 1e-6:
    #     lo = max(0.0, c_th - 0.02)
    #     hi = min(1.0, c_th + 0.02)
    #
    # fine_ths = np.round(np.linspace(lo, hi, int((hi - lo) / 0.001) + 1), 3)  # 例如 0.401, 0.402, ...
    # fine_records, (f_f1, f_th, f_p, f_r) = _sweep_thresholds(y_true_prob, y_prob, fine_ths)
    #
    # # 保存细搜 CSV
    # fine_csv = os.path.join(figures_dir, "threshold_sweep_metrics_fine.csv")
    # pd.DataFrame(fine_records).to_csv(fine_csv, index=False)
    # print(f"💾 细搜阈值扫描结果已保存: {fine_csv}")
    #
    # # 合并两阶段（去重按 threshold 排序后保存一份总表）
    # all_df = pd.concat([pd.DataFrame(coarse_records), pd.DataFrame(fine_records)], ignore_index=True)
    # all_df = all_df.drop_duplicates(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)
    # all_csv = os.path.join(figures_dir, "threshold_sweep_metrics_all.csv")
    # all_df.to_csv(all_csv, index=False)
    # print(f"💾 合并阈值扫描结果已保存: {all_csv}")
    #
    # # 绘图（使用合并后的数据）
    # plt.figure(figsize=(5, 4))
    # plt.plot(all_df["threshold"], all_df["f1"], label="F1")
    # plt.plot(all_df["threshold"], all_df["precision"], label="Precision")
    # plt.plot(all_df["threshold"], all_df["recall"], label="Recall")
    # plt.axvline(f_th, linestyle="--", label=f"best_th={f_th:.3f}")
    # plt.xlabel("Threshold");
    # plt.ylabel("Score");
    # plt.title("Metrics vs Threshold (Test)")
    # plt.grid(True, ls="--", alpha=0.4)
    # plt.legend()
    # plot_path = os.path.join(figures_dir, "metrics_vs_threshold.png")
    # plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    # plt.close()
    # print(f"🖼️ 指标-阈值曲线已保存: {plot_path}")
    #
    # # 最终报告“细搜”最佳阈值
    # print(f"🔧 Threshold tuning (two-stage) => best_th={f_th:.3f} | F1={f_f1:.4f} | P={f_p:.4f} | R={f_r:.4f}")
    # # ===== 以上为“先粗后细”阈值调优 =====


    # === 使用“验证集选出的 best_th”在测试集做一次固定阈值评估 ===
    best_th = best["th"]

    y_true_test_bin, y_prob_test = predict_proba(model, test_loader, device)
    y_pred_test_th = (y_prob_test >= best_th).astype(int)

    p_test = precision_score(y_true_test_bin, y_pred_test_th, zero_division=0)
    r_test = recall_score(y_true_test_bin, y_pred_test_th, zero_division=0)
    f1_test = f1_score(y_true_test_bin, y_pred_test_th)
    fpr_test = (lambda cm: cm[0,1] / (cm[0,1] + cm[0,0] + 1e-12))(confusion_matrix(y_true_test_bin, y_pred_test_th))

    print(f"📌 [TEST @th={best_th:.3f}] P={p_test:.4f} | R={r_test:.4f} | F1={f1_test:.4f} | FPR={fpr_test:.6f}")

    # 保存固定阈值下的测试集混淆矩阵
    cm_test = confusion_matrix(y_true_test_bin, y_pred_test_th)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix (Test, th={best_th:.3f})")
    plt.savefig(os.path.join(figures_dir, f"confusion_matrix_test_th{best_th:.3f}.png"),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"🖼️ 已保存测试混淆矩阵（固定阈值）：confusion_matrix_test_th{best_th:.3f}.png")


if __name__ == "__main__":
    main()
