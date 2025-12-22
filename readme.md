

## 📖 项目简介 (Introduction)

本项目为论文 **"基于CNN-GRU-Attention的DDoS攻击检测方法"** 的官方代码实现。

针对现有基于深度学习的DDoS检测方法在处理流量数据的空间局部性与时间依赖性方面存在的局限，本项目提出了一种名为 **DD-CGA** (Deep DDoS CNN-GRU-Attention) 的混合神经网络架构。该方法创新性地将网络流特征重构为二维灰度图，利用 **CNN** 提取局部空间特征，引入 **CBAM (Convolutional Block Attention Module)** 增强关键特征的显著性；随后通过多种序列化策略（如Z-order, Hilbert曲线）将特征图展平，利用 **Bi-GRU** 捕捉长距离时序依赖；最后采用 **Attention Pooling** 机制动态聚合时序特征，实现对DDoS攻击的高精度检测。

### ✨ 核心特性 (Key Features)

* **多维度特征融合**：结合 CNN 的空间特征提取能力与 GRU 的时序建模能力。
* **双重注意力机制**：
* **前端**：集成 CBAM (Channel + Spatial Attention) 抑制背景噪声。
* **后端**：实现 Additive/Scaled-Dot Attention Pooling，缓解时序信息的“遗忘”问题。


* **鲁棒的评估体系**：
* 包含两阶段（粗搜+细搜）阈值自适应调优策略 (`tune_threshold_and_eval.py`)。
* 集成 **Bootstrap 置信区间**计算 (`bootstrap_ci_gru.py`)，确保实验结果具有统计显著性。


* **灵活的配置管理**：基于 YAML 文件的全参数化配置，支持一键运行消融实验。

## 🏗️ 系统架构 (Architecture)

模型整体处理流程如下：

1. **数据预处理**：CSV流量特征  归一化  8x8 特征矩阵。
2. **空间特征提取**：双层 CNN + CBAM 注意力模块。
3. **序列化**：支持 Row-major / Z-order / Hilbert 曲线扫描。
4. **时序建模**：Bidirectional GRU。
5. **特征聚合与分类**：Attention Pooling  MLP  Sigmoid。

## 📂 目录结构 (Directory Structure)

```text
DD-CGA/
├── config/                 # 实验配置文件 (YAML)
│   ├── cnn_baseline.py     # 基线模型配置
│   ├── cnn_gru.py          # CNN-GRU 消融配置
│   └── cnn_gru_att.yaml    # DD-CGA 完整模型配置
├── datas/                  # 数据存放目录 (需自行准备或通过脚本生成)
│   └── splits/             # 训练/验证/测试集划分
├── models/                 # 模型定义
│   ├── cnn_baseline.py     # CNN 基线
│   ├── cnn_gru.py          # CNN + GRU
│   └── cnn_gru_attn.py     # CNN + CBAM + GRU + Attention Pooling (核心)
├── pre_data_final/         # 数据预处理流水线脚本
├── results/                # 实验结果输出 (日志, 权重, 图表, 统计数据)
├── tools/                  # 评估与分析工具箱
│   ├── run_ablation.py     # 一键运行消融实验
│   ├── bootstrap_ci_gru.py # 计算置信区间
│   ├── tune_threshold.py   # 阈值调优
│   └── ...
├── training/               # 训练脚本
│   ├── train.py            # 通用训练入口
│   ├── dataset_loader.py   # 数据加载器 (8x8 Reshape逻辑)
│   └── ...
└── README.md

```

## 🚀 快速开始 (Getting Started)

### 1. 环境依赖 (Prerequisites)

请确保安装 Python 3.8+ 及 PyTorch。

```bash
pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn pyyaml tqdm

```

### 2. 数据准备 (Data Preparation)

本项目使用 CIC-IDS2017 / CIC-DDoS2019 等通用数据集格式。请按以下顺序执行脚本完成数据清洗与划分：

```bash
# 1. 流量筛选与合并
python pre_data_final/select_normal.py
python pre_data_final/select_attacks.py
python pre_data_final/combine_fin_csvs.py

# 2. 特征工程与编码
python pre_data_final/feature_analysis_remove_columns.py # 剔除无效列
python pre_data_final/protocol_onehot.py                 # 协议独热编码

# 3. 采样与数据集划分
python pre_data_final/select_sample_2M.py                # 均衡采样
python pre_data_final/split_dataset_2M.py                # 划分 Train/Val/Test

```

*注：处理后的 CSV 文件应位于 `datas/splits/` 目录下。*

### 3. 模型训练 (Training)

#### 训练完整模型 (DD-CGA)

使用 `training/train_cnn_gru_att.py` 脚本，通过指定配置文件启动训练：

```bash
python training/train_cnn_gru_att.py --config config/cnn_gru_att.yaml

```

#### 训练消融实验变体

如需对比 CNN 基线或无 Attention 的 GRU 模型：

```bash
# CNN Baseline
python training/train.py --config config/config.yaml

# CNN + GRU (无 Attention)
python training/train_cnn_gru.py --config config/cnn_gru.yaml

```

### 4. 评估与推理 (Evaluation & Inference)

#### 阈值自适应调优

模型训练完成后，使用验证集寻找最佳 F1 阈值，并在测试集上进行最终评估：

```bash
python tools/tune_threshold_and_eval_gru.py \
  --config config/cnn_gru_att.yaml \
  --ckpt results/checkpoints/<timestamp>/checkpoint_best.pt

```

#### 统计显著性分析 (Bootstrap CI)

为了验证模型改进的有效性，使用 Bootstrap 方法计算指标的 95% 置信区间：

```bash
python tools/bootstrap_ci_gru.py \
  --preds_a results/tuning/baseline/test_preds.csv \
  --preds_b results/tuning/dd_cga/test_preds.csv \
  --metric f1 \
  --paired true \
  --n_boot 10000 \
  --out results/ci/comparison_result.json

```

## 📊 实验结果 (Results)

*以下数据基于 CIC-DDoS2019 数据集测试结果 (示例)*

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time (ms) |
| --- | --- | --- | --- | --- | --- |
| CNN Baseline | 98.12% | 98.05% | 98.20% | 98.12% | **0.85** |
| CNN-GRU | 99.35% | 99.10% | 99.60% | 99.35% | 1.20 |
| **DD-CGA (Ours)** | **99.87%** | **99.85%** | **99.90%** | **99.87%** | 1.45 |

详细的消融实验结果（包括序列顺序的影响、CBAM的有效性分析）请参阅 `results/ablation/` 目录下的汇总报表。

## 🛠️ 工具脚本说明

* **`tools/error_overlap.py`**: 分析不同模型（如 CNN 与 GRU）错误样本的重叠度，证明模型间的互补性。
* **`tools/run_attn_ablation_suite.py`**: 一键运行 Attention 机制的三组对比实验（ab1/ab2/ab3）。
* **`tools/plot_threshold_eval.py`**: 绘制 F1-Threshold 曲线、PR 曲线与 ROC 曲线。

