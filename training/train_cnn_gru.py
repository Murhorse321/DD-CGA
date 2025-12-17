# training/train_cnn_gru.py
"""
CNN+GRU 训练脚本（独立于 baseline 的 train.py）
- 读取 YAML 配置（--config 指定，默认 config/cnn_gru.yaml）
- 日志/权重/图表均写入带时间戳的目录，不会覆盖旧实验
- 训练/验证/早停/测试完整流程
- PR 曲线、混淆矩阵图片
- 固定阈值(threshold from cfg)与最优F1阈值比较
- 自动保存 summary.json 便于对比实验汇总
"""
import os
import time
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt

# 你项目里的组件
from training.dataset_loader import get_dataloaders
from models.cnn_gru import CNNGRU  # ← 只用 CNN+GRU（不含 Attention）


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm: float = 1.0):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        # 防梯度爆炸（RNN 常见）
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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


def main(args):
    # 1) 读取配置
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 通用配置
    th_cfg = float(cfg.get("inference", {}).get("threshold", 0.5))

    # 训练超参
    lr       = float(cfg["training"].get("learning_rate", 1e-3))
    epochs   = int(cfg["training"].get("epochs", 20))
    wd       = float(cfg["training"].get("weight_decay", 0.0))
    patience = int(cfg["training"].get("early_stopping_patience", 5))
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0)) if "label_smoothing" in cfg["training"] else 0.0
    best_on  = cfg["training"].get("best_on", "val_f1_macro")  # 'val_acc' or 'val_f1_macro'
    base_log_dir  = cfg["training"].get("log_dir", "results/logs/cnn_gru")
    base_ckpt_dir = cfg["training"].get("ckpt_dir", "results/checkpoints")

    # 2) 生成时间戳目录
    timestamp  = time.strftime("%Y%m%d-%H%M%S")
    log_dir    = f"{base_log_dir}_{timestamp}"
    ckpt_dir   = os.path.join(base_ckpt_dir, timestamp)
    figures_dir = os.path.join("results/figures", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 3) 固定随机种子 & 设备
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")
    print(f"▶ LogDir:   {log_dir}")
    print(f"▶ CkptDir:  {ckpt_dir}")
    print(f"▶ Figures:  {figures_dir}")

    # 4) DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(config_path)

    # 5) 模型/损失/优化器/调度
    model_params = cfg.get("model", {}).get("params", {})
    model = CNNGRU(num_classes=2, **model_params).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )

    # 6) TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 7) 早停
    best_metric = -1.0
    best_ckpt = os.path.join(ckpt_dir, "cnn_gru_best.pth")
    wait = 0

    # 8) 训练循环
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, val_acc, y_true_val, y_pred_val = evaluate(model, val_loader, criterion, device)
        val_f1_macro = f1_score(y_true_val, y_pred_val, average="macro")
        val_f1_weighted = f1_score(y_true_val, y_pred_val, average="weighted")

        # TB 记录
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val",   val_loss,   epoch)
        writer.add_scalar("Acc/Train",  train_acc,  epoch)
        writer.add_scalar("Acc/Val",    val_acc,    epoch)
        writer.add_scalar("F1/Val_macro",    val_f1_macro, epoch)
        writer.add_scalar("F1/Val_weighted", val_f1_weighted, epoch)

        # LR 调度（监控 val_acc，更稳）
        scheduler.step(val_acc)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] "
              f"Train Loss  {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val  Loss   {val_loss:.4f} Acc {val_acc:.4f} F1(macro) {val_f1_macro:.4f} | "
              f"{dt:.1f}s")

        # 保存最优
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

    # 9) 加载最优权重
    try:
        state_dict = torch.load(best_ckpt, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print(f"\n✅ Loaded best checkpoint: {best_ckpt}")

    # 10) 测试集评估
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    test_f1_macro = f1_score(y_true, y_pred, average="macro")
    test_f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"\n🧪 Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
          f"Test F1(macro): {test_f1_macro:.4f} | Test F1(weighted): {test_f1_weighted:.4f}")
    print("\n📋 Classification Report (Test):")
    print(classification_report(y_true, y_pred, digits=4))

    # 11) AUC/PR-AUC + 图表
    y_true_prob, y_prob = predict_proba(model, test_loader, device)
    auc = roc_auc_score(y_true_prob, y_prob)
    prec, rec, _ = precision_recall_curve(y_true_prob, y_prob)
    ap = average_precision_score(y_true_prob, y_prob)
    print(f"🔵 Test ROC-AUC: {auc:.4f}")
    print(f"🟣 Test PR-AUC (Average Precision): {ap:.4f}")

    # 混淆矩阵（自然阈值 = argmax）
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix (Test, argmax)")
    plt.savefig(os.path.join(figures_dir, "confusion_matrix_test_argmax.png"),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

    # PR 曲线
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (Test)")
    plt.grid(True, ls='--', alpha=0.4)
    plt.savefig(os.path.join(figures_dir, "pr_curve_test.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # 12) 用 cfg 阈值再评估一次
    y_pred_cfg = (y_prob >= th_cfg).astype(int)
    p_cfg = precision_score(y_true_prob, y_pred_cfg)
    r_cfg = recall_score(y_true_prob, y_pred_cfg)
    f1_cfg = f1_score(y_true_prob, y_pred_cfg)
    print(f"📌 Using cfg threshold={th_cfg:.3f} => P={p_cfg:.4f} R={r_cfg:.4f} F1={f1_cfg:.4f}")

    cm_cfg = confusion_matrix(y_true_prob, y_pred_cfg)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_cfg)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix (Test, th={th_cfg:.2f})")
    plt.savefig(os.path.join(figures_dir, f"confusion_matrix_test_th{th_cfg:.2f}.png"),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 13) 阈值调优：找到最优F1阈值
    best_f1, best_th = -1.0, 0.5
    for th in np.linspace(0.1, 0.9, 33):
        y_hat = (y_prob >= th).astype(int)
        f1v = f1_score(y_true_prob, y_hat)
        if f1v > best_f1:
            best_f1, best_th = f1v, th
    y_hat_best = (y_prob >= best_th).astype(int)
    p_best = precision_score(y_true_prob, y_hat_best)
    r_best = recall_score(y_true_prob, y_hat_best)
    print(f"🔧 Threshold tuning => best_th={best_th:.3f} | F1={best_f1:.4f} | P={p_best:.4f} | R={r_best:.4f}")

    # 14) 保存 summary.json（便于后续比较/画表）
    summary = {
        "model": "cnn_gru",
        "params": model_params,
        "best_on": best_on,
        "paths": {
            "log_dir": log_dir,
            "ckpt": best_ckpt,
            "figures_dir": figures_dir,
            "config": config_path,
        },
        "test": {
            "loss": float(test_loss),
            "acc": float(test_acc),
            "f1_macro": float(test_f1_macro),
            "f1_weighted": float(test_f1_weighted),
            "roc_auc": float(auc),
            "pr_auc": float(ap),
        },
        "threshold_cfg": float(th_cfg),
        "threshold_cfg_metrics": {
            "precision": float(p_cfg), "recall": float(r_cfg), "f1": float(f1_cfg)
        },
        "threshold_best": float(best_th),
        "threshold_best_metrics": {
            "precision": float(p_best), "recall": float(r_best), "f1": float(best_f1)
        },
        "time": timestamp,
    }
    with open(os.path.join(figures_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📝 Saved summary to {os.path.join(figures_dir, 'summary.json')}")

    print(f"🖼️ 图表与摘要已保存到：{figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/cnn_gru.yaml",
                        help="Path to YAML config for CNN+GRU")
    args = parser.parse_args()
    main(args)
