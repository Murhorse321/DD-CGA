# training/train.py
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

# 项目组件
from training.dataset_loader import get_dataloaders
from models.cnn_baseline import CNNBaseline


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证实验可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
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
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        # 获取正类概率 (Class 1)
        p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(p1)
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main(args):
    # 1. 读取配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 实验参数
    seed = int(cfg.get("seed", 42))  # [修改] 从config读取seed
    set_seed(seed)

    # 路径参数
    base_log_dir = cfg["training"].get("log_dir", "results/logs/cnn_baseline")
    base_ckpt_dir = cfg["training"].get("ckpt_dir", "results/checkpoints/cnn_baseline")

    # 生成时间戳目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{base_log_dir}_{timestamp}"
    ckpt_dir = os.path.join(base_ckpt_dir, timestamp)
    figures_dir = os.path.join("results/cnn_baseline/figures", timestamp)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 备份本次实验的 Config 到输出目录（而不是修改原文件）
    with open(os.path.join(figures_dir, "config_backup.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")
    print(f"▶ Output: {figures_dir}")

    # 3. 数据加载
    train_loader, val_loader, test_loader = get_dataloaders(args.config)

    # 4. 模型构建
    model_params = cfg.get("model", {}).get("params", {})
    num_classes = cfg.get("model", {}).get("num_classes", 2)
    dropout = float(cfg["training"].get("dropout", 0.5))

    # 实例化模型
    model = CNNBaseline(num_classes=num_classes, dropout=dropout).to(device)

    # 优化器
    lr = float(cfg["training"].get("learning_rate", 1e-3))
    wd = float(cfg["training"].get("weight_decay", 0.0))
    epochs = int(cfg["training"].get("epochs", 20))
    patience = int(cfg["training"].get("early_stopping_patience", 5))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    writer = SummaryWriter(log_dir=log_dir)
    best_metric = -1.0
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    wait = 0
    best_on = cfg["training"].get("best_on", "val_acc")

    # 5. 训练循环
    print("\n🚀 Start Training...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, y_true_val, y_pred_val = evaluate(model, val_loader, criterion, device)

        val_f1 = f1_score(y_true_val, y_pred_val, average="macro")

        # 记录日志
        writer.add_scalar("Loss/Train", tr_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Train", tr_acc, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(val_acc)

        dt = time.time() - t0
        print(
            f"[Ep {epoch:02d}] Tr_Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | Val_Loss:{val_loss:.4f} Acc:{val_acc:.4f} F1:{val_f1:.4f} | {dt:.1f}s")

        # 早停检查
        current_metric = val_acc if best_on == "val_acc" else val_f1
        if current_metric > best_metric:
            best_metric = current_metric
            wait = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"   ✅ Best {best_on} updated: {best_metric:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"   🛑 Early stopping at epoch {epoch}")
                break

    writer.close()

    # 6. 最终评估 (使用 Best Ckpt)
    print("\n🧪 Start Testing (Loading Best Checkpoint)...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # 在测试集上跑基础指标 (Argmax)
    test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, criterion, device)
    test_f1 = f1_score(y_true_test, y_pred_test, average="macro")
    print(f"   Test Acc (Argmax): {test_acc:.4f} | F1 (Macro): {test_f1:.4f}")
    print(classification_report(y_true_test, y_pred_test, digits=4))

    # 7. 阈值分析 (Threshold Analysis)
    # 基于验证集寻找最佳阈值 (Threshold Tuning on Validation)
    print("\n🔧 Tuning Threshold on Validation Set...")
    y_true_val_bin, y_prob_val = predict_proba(model, val_loader, device)

    # 策略：寻找最大化 F1 的阈值
    best_th_val = 0.5
    best_f1_val = 0.0
    for th in np.linspace(0.01, 0.99, 99):
        y_hat = (y_prob_val >= th).astype(int)
        f1_v = f1_score(y_true_val_bin, y_hat)
        if f1_v > best_f1_val:
            best_f1_val = f1_v
            best_th_val = th

    print(f"   Best Threshold (from Val): {best_th_val:.3f} (Val F1: {best_f1_val:.4f})")

    # 应用该阈值到测试集
    y_true_test_bin, y_prob_test = predict_proba(model, test_loader, device)
    y_pred_test_tuned = (y_prob_test >= best_th_val).astype(int)

    test_f1_tuned = f1_score(y_true_test_bin, y_pred_test_tuned)
    test_prec_tuned = precision_score(y_true_test_bin, y_pred_test_tuned)
    test_rec_tuned = recall_score(y_true_test_bin, y_pred_test_tuned)

    print(f"   [Test @ Best Val Th] F1: {test_f1_tuned:.4f} | P: {test_prec_tuned:.4f} | R: {test_rec_tuned:.4f}")

    # 8. 保存结果与图表
    # 混淆矩阵
    cm = confusion_matrix(y_true_test_bin, y_pred_test_tuned)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix (Th={best_th_val:.3f})")
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # AUC / PR Curve
    auc = roc_auc_score(y_true_test_bin, y_prob_test)
    ap = average_precision_score(y_true_test_bin, y_prob_test)

    summary = {
        "timestamp": timestamp,
        "config": args.config,
        "best_epoch": int(epochs - wait),
        "best_val_metric": float(best_metric),
        "test_argmax": {"acc": float(test_acc), "f1_macro": float(test_f1)},
        "tuned_threshold": {
            "best_th_val": float(best_th_val),
            "test_f1": float(test_f1_tuned),
            "test_precision": float(test_prec_tuned),
            "test_recall": float(test_rec_tuned),
            "test_auc": float(auc),
            "test_pr_auc": float(ap)
        }
    }

    with open(os.path.join(figures_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Results saved to: {figures_dir}")
    # 注意：这里删除了所有回写 config.yaml 的代码


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args)