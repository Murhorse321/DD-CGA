# training/dataset_loader.py
# -*- coding: utf-8 -*-
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CSVDataset(Dataset):
    def __init__(self, csv_file, H=8, W=8, label_col='label_int', ignore_cols=None):
        """
        Args:
            label_col: 标签列名，必须是数值型的标签 (例如 'label_int')
            ignore_cols: 需要排除的非特征列 (例如字符串类型的 'label')
        """
        # 默认忽略的列（适配你的 Step 6 产出）
        if ignore_cols is None:
            ignore_cols = ['label', 'label_int']

            # 1. 读取数据
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"数据文件不存在: {csv_file}")

        df = pd.read_csv(csv_file)

        # 2. 确定标签 Y
        if label_col not in df.columns:
            # 尝试回退：如果你配置写了 'label' 但只有 'label_int'，或者反之
            raise KeyError(f"CSV 中找不到标签列 '{label_col}'。现有列: {list(df.columns)}")

        y = df[label_col].values.astype(np.int64)

        # 3. 确定特征 X
        # 逻辑：特征列 = 所有列 - 标签列 - 忽略列
        # 这样可以安全地把 'label' (字符串) 排除在特征之外
        cols_to_exclude = set(ignore_cols) | {label_col}
        feature_cols = [c for c in df.columns if c not in cols_to_exclude]

        # 4. 特征维度检查与处理
        target_dim = H * W
        actual_dim = len(feature_cols)

        # 打印调试信息 (仅在第一次初始化时打印，避免刷屏)
        # 这里简单打印一下特征数，确保是 64
        # print(f"加载文件: {os.path.basename(csv_file)} | 特征数: {actual_dim} | 标签列: {label_col}")

        X = df[feature_cols].values.astype(np.float32)

        # 维度对齐逻辑
        if actual_dim > target_dim:
            # 截断（虽然 Step 6 已经保证了 64，但保留此逻辑以防万一）
            X = X[:, :target_dim]
        elif actual_dim < target_dim:
            # 补零
            pad = np.zeros((X.shape[0], target_dim - actual_dim), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)

        # 5. Reshape [N, 1, 8, 8]
        X = X.reshape(-1, 1, H, W)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)


def _as_cfg(config_path_or_dict):
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    with open(config_path_or_dict, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_paths(data_cfg):
    # 优先读取 normalized 的路径，如果配置没改，这里需要手动注意
    train = data_cfg.get("train_path")
    val = data_cfg.get("val_path")
    test = data_cfg.get("test_path")
    if not (train and val and test):
        raise KeyError("Config 中缺少 train_path / val_path / test_path 配置")
    return train, val, test


def get_dataloaders(config_path_or_dict):
    cfg = _as_cfg(config_path_or_dict)

    # 获取参数
    bs = int(cfg.get("training", {}).get("batch_size", 512))
    data_cfg = cfg.get("data", {})

    # 获取维度
    shape = data_cfg.get("input_shape", [8, 8])
    H, W = int(shape[0]), int(shape[1])

    # 关键修改：默认使用 'label_int' 作为标签列
    # 同时在 dataset 内部会自动忽略 'label' 字符串列
    label_col = data_cfg.get("label_col", "label_int")

    train_path, val_path, test_path = _get_paths(data_cfg)

    def _dl(path, shuffle):
        # 传入 ignore_cols=['label'] 以排除字符串列
        ds = CSVDataset(path, H, W, label_col=label_col, ignore_cols = data_cfg.get("ignore_cols", []))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = _dl(train_path, True)
    val_loader = _dl(val_path, False)
    test_loader = _dl(test_path, False)

    return train_loader, val_loader, test_loader