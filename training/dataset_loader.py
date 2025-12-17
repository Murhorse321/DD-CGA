# training/dataset_loader.py
# -*- coding: utf-8 -*-
"""
兼容改造版：
- with open(..., encoding='utf-8') 避免 GBK 报错
- get_dataloaders() 既接受 YAML 路径，也接受已解析的 dict
- 兼容键名：
    data.input_shape 或 data.image_size
    data.train_path/val_path/test_path 或 data.train_csv/val_csv/test_csv
- CSV 要求：必须有 label 列；其余列视为特征。自动截断/补零到 H*W，并 reshape -> [B,1,H,W]
"""

import os, yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, csv_file, H=8, W=8, label_col='label', feature_cols=None):
        df = pd.read_csv(csv_file)
        if label_col not in df.columns:
            raise KeyError(f"CSV 缺少标签列 '{label_col}'，实际列：{list(df.columns)[:6]} ...")
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)

        # 自动对齐到 H*W
        need = H*W
        have = X.shape[1]
        if have > need:
            X = X[:, :need]
        elif have < need:
            pad = np.zeros((X.shape[0], need - have), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)

        X = X.reshape(-1, 1, H, W)
        self.X = X
        self.y = y

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

def _as_cfg(config_path_or_dict):
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    with open(config_path_or_dict, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _get_shape(data_cfg):
    shape = data_cfg.get("input_shape") or data_cfg.get("image_size")
    if not shape or len(shape) != 2:
        raise KeyError("请在 data.input_shape 或 data.image_size 中提供 [H, W]，例如 [8, 8]")
    H, W = int(shape[0]), int(shape[1])
    return H, W

def _get_paths(data_cfg):
    train = data_cfg.get("train_path") or data_cfg.get("train_csv")
    val   = data_cfg.get("val_path")   or data_cfg.get("val_csv")
    test  = data_cfg.get("test_path")  or data_cfg.get("test_csv")
    if not (train and val and test):
        raise KeyError("请在 data 下提供 train_path/val_path/test_path（或 train_csv/val_csv/test_csv）")
    return train, val, test

def get_dataloaders(config_path_or_dict):
    cfg = _as_cfg(config_path_or_dict)
    bs = int(cfg.get("training", {}).get("batch_size", 512))
    data_cfg = cfg.get("data", {})
    H, W = _get_shape(data_cfg)
    label_col = data_cfg.get("label_col", "label")
    feature_cols = data_cfg.get("feature_cols", None)
    train_path, val_path, test_path = _get_paths(data_cfg)

    def _dl(path, shuffle):
        ds = CSVDataset(path, H, W, label_col=label_col, feature_cols=feature_cols)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = _dl(train_path, True)
    val_loader   = _dl(val_path, False)
    test_loader  = _dl(test_path, False)
    return train_loader, val_loader, test_loader
