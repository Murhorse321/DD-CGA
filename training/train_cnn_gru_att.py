# training/train_cnn_gru_att.py
# -*- coding: utf-8 -*-
"""
独立的 CNN+GRU+Attention 训练脚本（不会影响你原有 train_cnn_gru.py）
修复点：
- 读取 config 中的 model.num_classes（默认 1），与 BCEWithLogitsLoss 对齐
- 自适配你们的 dataset_loader.get_dataloaders：既可传 YAML 路径，也可传 dict
- 显式 UTF-8、AMP 新 API（消除 FutureWarning）
- 训练结束导出：val_probs.csv / test_probs.csv / test_preds.csv / summary.json
"""

import os, sys, time, json, yaml, math, random
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== 实用函数 ==========
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device(cfg):
    dev = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if dev == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，改用 CPU")
        dev = 'cpu'
    return torch.device(dev)

# ========== 兼容导入 ==========
def _try_import_loader():
    try:
        from training.dataset_loader import get_dataloaders  # 你们项目里的
        return get_dataloaders
    except Exception:
        try:
            from dataset_loader import get_dataloaders       # 兼容根目录
            return get_dataloaders
        except Exception:
            return None

def import_model():
    try:
        from models.cnn_gru_attn import CNNGRUAttn
    except Exception:
        from cnn_gru_attn import CNNGRUAttn
    return CNNGRUAttn

# ========== 兜底 CSV Dataset（若没有你们的 loader 时使用） ==========
class CSVDataset(Dataset):
    def __init__(self, csv_file, image_size=(8,8), label_col="label", feature_cols=None):
        import pandas as pd
        df = pd.read_csv(csv_file)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)
        H,W = image_size
        assert X.shape[1] == H*W, f"期望 {H*W} 特征，得到 {X.shape[1]}"
        X = X.reshape(-1,1,H,W)
        self.X = X; self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

def default_get_dataloaders(cfg):
    bs = int(cfg['training'].get('batch_size', 512))
    H,W = cfg['data'].get('image_size', [8,8])
    label_col = cfg['data'].get('label_col','label')
    feature_cols = cfg['data'].get('feature_cols', None)
    def _dl(path, shuffle):
        ds = CSVDataset(path, (H,W), label_col, feature_cols)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)
    train_loader = _dl(cfg['data']['train_csv'], True)
    val_loader   = _dl(cfg['data']['val_csv'],   False)
    test_loader  = _dl(cfg['data']['test_csv'],  False)
    return train_loader, val_loader, test_loader

# ========== 指标 ==========
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_probs(y_true, y_prob, th=0.5):
    y_pred = (y_prob >= th).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(th),
    }

def sweep_best_f1(y_true, y_prob, lo=0.05, hi=0.99, n=991):
    ths = np.linspace(lo, hi, n)
    best = (0.0, 0.5)
    for th in ths:
        f1 = f1_score(y_true, (y_prob >= th).astype(int))
        if f1 > best[0]: best = (f1, float(th))
    return {"best_f1": best[0], "best_th": best[1]}

# ========== 训练与推理 ==========
from torch import amp as torch_amp  # 新 AMP API

def train_one_epoch(model, loader, device, optimizer, scaler=None, criterion=None):
    model.train()
    total, loss_sum = 0, 0.0
    autocast_device = device.type if device.type in ('cuda','cpu') else 'cpu'
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()  # BCE -> float
        optimizer.zero_grad(set_to_none=True)
        with torch_amp.autocast(autocast_device):
            logits = model(xb)                           # [B,1] 或 [B]
            loss = criterion(logits.squeeze(-1), yb)     # 与 BCE 对齐
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = xb.size(0)
        total += bs
        loss_sum += float(loss.item()) * bs
    return loss_sum / max(1,total)

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        logits = model(xb).squeeze(-1)       # [B]
        p = torch.sigmoid(logits)            # BCE pipeline
        probs.append(p.detach().cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)

# ========== 主流程 ==========
def main(args):
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    set_seed(int(cfg.get('seed', 42)))
    device = get_device(cfg)

    # 结果目录
    time_tag = time.strftime("%Y%m%d-%H%M%S")
    run_root = cfg['training'].get('run_root', "results/tuning_gru_attn")
    run_dir = Path(run_root) / f"ATT_{time_tag}"
    ckpt_dir = run_dir / "ckpt"; fig_dir = run_dir / "fig"
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)

    # 备份配置
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # DataLoaders：优先用你们的 loader；既尝试传 dict，也尝试传路径；失败则用兜底 CSV
    get_dl = _try_import_loader()
    if get_dl is not None:
        try:
            train_loader, val_loader, test_loader = get_dl(cfg)          # 一些版本支持 dict
        except TypeError:
            train_loader, val_loader, test_loader = get_dl(args.config)  # 你的版本多半是路径
    else:
        train_loader, val_loader, test_loader = default_get_dataloaders(cfg)

    # 模型
    CNNGRUAttn = import_model()
    mcfg = cfg['model']
    num_classes = int(mcfg.get('num_classes', 1))  # 默认单通道+BCE
    model = CNNGRUAttn(
        num_classes=num_classes,
        cnn_channels=tuple(mcfg.get('cnn_channels', [32,64])),
        use_cbam=bool(mcfg.get('use_cbam', True)),
        cbam_reduction=int(mcfg.get('cbam_reduction', 8)),
        gru_hidden=int(mcfg.get('gru_hidden', 128)),
        gru_layers=int(mcfg.get('gru_layers', 1)),
        bidirectional=bool(mcfg.get('bidirectional', False)),
        attn_type=str(mcfg.get('attn_type', 'add')),
        dropout=float(mcfg.get('dropout', 0.5)),
        use_batchnorm=bool(mcfg.get('use_batchnorm', True)),
        sequence_order=str(mcfg.get('sequence_order', 'row')),
        temperature=float(mcfg.get('temperature', 1.0)),
    ).to(device)

    # 训练配置
    tcfg = cfg['training']
    epochs   = int(tcfg.get('epochs', 20))
    lr       = float(tcfg.get('learning_rate', 5e-4))
    wd       = float(tcfg.get('weight_decay', 1e-4))
    pos_w    = float(tcfg.get('pos_weight', 1.0))
    use_amp  = bool(tcfg.get('amp', True))
    patience = int(tcfg.get('early_stopping_patience', 8))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = None
    if tcfg.get('scheduler','none').lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif tcfg.get('scheduler','none').lower() == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(tcfg.get('step_size', 10)), gamma=float(tcfg.get('gamma', 0.1)))

    # BCE（与单通道输出对齐）
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    scaler = torch_amp.GradScaler(device.type if device.type in ('cuda','cpu') else 'cpu',
                                  enabled=(use_amp and device.type=='cuda'))

    # 训练循环 + 早停
    best = {"val_f1": -1.0, "epoch": -1}
    stale = 0
    for epoch in range(1, epochs+1):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, scaler, criterion)

        # 验证
        val_prob, val_true = predict_probs(model, val_loader, device)
        swp = sweep_best_f1(val_true, val_prob )
        val_metrics = evaluate_probs(val_true, val_prob, th=swp['best_th'])

        msg = (f"[{epoch:03d}/{epochs}] train_loss={tr_loss:.5f} | "
               f"val_f1={val_metrics['f1']:.4f} (th*={swp['best_th']:.3f}) "
               f"val_acc={val_metrics['acc']:.4f} prAUC={val_metrics['pr_auc']:.4f} rocAUC={val_metrics['roc_auc']:.4f}")
        print(msg)

        # 保存最佳
        if val_metrics['f1'] > best['val_f1'] + 1e-6:
            best.update(val_f1=val_metrics['f1'], epoch=epoch, th_best=float(swp['best_th']))
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_dir / "checkpoint_best.pt")
            import pandas as pd
            pd.DataFrame({"y_true": val_true, "y_prob": val_prob}).to_csv(run_dir / "val_probs.csv", index=False)
            stale = 0
        else:
            stale += 1

        if scheduler is not None: scheduler.step()
        if stale >= patience:
            print(f"⏹️ Early stopping at epoch {epoch} (best epoch={best['epoch']})"); break

    # 载入最佳并在 Test 上评估
    if (ckpt_dir / "checkpoint_best.pt").exists():
        state = torch.load(ckpt_dir / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(state["model"])

    test_prob, test_true = predict_probs(model, test_loader, device)

    import pandas as pd
    pd.DataFrame({"y_true": test_true, "y_prob": test_prob}).to_csv(run_dir / "test_probs.csv", index=False)

    th_cfg = float(cfg.get('inference', {}).get('threshold', 0.5))
    th_star = float(best.get('th_best', th_cfg))
    test_at_star = evaluate_probs(test_true, test_prob, th_star)
    test_at_cfg  = evaluate_probs(test_true, test_prob, th_cfg)
    print(f"[TEST] @th*={th_star:.3f}  F1={test_at_star['f1']:.4f}  P={test_at_star['precision']:.4f}  R={test_at_star['recall']:.4f}  "
          f"ROC={test_at_star['roc_auc']:.4f} PR={test_at_star['pr_auc']:.4f}")
    print(f"[TEST] @cfg={th_cfg:.3f}  F1={test_at_cfg['f1']:.4f}  P={test_at_cfg['precision']:.4f}  R={test_at_cfg['recall']:.4f}")

    y_pred = (test_prob >= th_star).astype(int)
    pd.DataFrame({"y_true": test_true, "y_prob": test_prob, "y_pred": y_pred}).to_csv(run_dir / "test_preds.csv", index=False)

    summary = {
        "epoch_best": int(best.get('epoch', -1)),
        "val_f1_best": float(best.get('val_f1', 0.0)),
        "val_best_th": float(th_star),
        "test_at_best": test_at_star,
        "test_at_cfg":  test_at_cfg,
        "run_dir": str(run_dir),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ 训练完成 | 结果目录：{run_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/cnn_gru_att.yaml")
    args = ap.parse_args()
    main(args)
