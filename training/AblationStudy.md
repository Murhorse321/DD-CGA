
# CNN–GRU–Attention 消融实验完整方案（面向 DDoS 检测）

> 本消融实验方案旨在**系统性验证各模块（CNN、GRU、Attention）在 DDoS 攻击检测任务中的贡献**，形成一条**从简单基线到完整模型的清晰技术演进路径**。  
> 该方案可直接作为论文 **“Ablation Study / 消融实验”** 小节的实验依据。

---

## 一、实验总体目标

通过逐步引入时序建模与注意力机制，回答以下核心研究问题：

1. **仅使用 CNN 的空间特征建模是否足以区分 DDoS 与正常流量？**
2. **在 CNN 基础上引入 GRU 进行序列建模是否能提升检测性能？**
3. **Attention 机制是否能够进一步强化关键 token 的判别能力？**
4. **不同 Attention 形式与 token 顺序对模型性能有何影响？**

---

## 二、数据与通用实验设置（所有实验统一）

### 2.1 数据集

- 数据集：**CIC-DDoS2019**
- 特征数：64（经清洗、归一化后）
- 输入形式：`8 × 8` 单通道特征图
- 任务类型：二分类（Normal vs DDoS）

### 2.2 数据划分

| 数据集 | 用途 |
|------|------|
| Train | 参数学习 |
| Validation | 模型选择 + 阈值选择 |
| Test | 最终性能评估 |

> ⚠️ 测试集**不参与任何训练或阈值搜索**

---

### 2.3 统一训练配置（保持不变）

| 项目 | 设置 |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Batch Size | 512 |
| Epochs | 100 |
| Early Stopping | patience = 10 |
| 学习率 | 1e-3 |
| AMP | 启用 |
| 指标 | Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC |

---

## 三、模型消融路径总览（核心）

> 消融顺序遵循 **“能力最小 → 结构增强”** 原则

```

M0: CNN-only
↓
M1: CNN + GRU
↓
M2: CNN + GRU + Attention（Additive）
↓
M3: CNN + GRU + Attention（Dot）

```

---

## 四、阶段一：CNN-only 基线模型（M0）

### 4.1 研究目的

验证 **仅利用空间相关性（不建模 token 依赖）** 的检测能力，为后续模型提供性能下界。

---

### 4.2 模型结构

```

Input (1×8×8)
→ CNN
→ Global Average Pooling
→ FC
→ Sigmoid

````

#### 关键说明
- 不使用 GRU
- 不使用 Attention
- CNN 后直接池化为全局特征

---

### 4.3 配置示例

```yaml
model:
  backbone: cnn
  use_gru: false
  use_attention: false
````

---

### 4.4 记录指标

* Val / Test：

  * Accuracy
  * F1-score
  * ROC-AUC
  * PR-AUC

---

### 4.5 预期结果（论文可用语言）

> CNN-only 模型能够捕获局部特征分布，但由于忽略特征间的结构依赖，其整体检测性能受限。

---

## 五、阶段二：CNN + GRU（M1）

### 5.1 研究目的

验证 **引入序列建模机制（GRU）是否能提升检测性能**。

---

### 5.2 模型结构

```
Input
 → CNN
 → Tokenization (4×4 → 16 tokens)
 → GRU
 → Last hidden state
 → FC
 → Sigmoid
```

---

### 5.3 Token 构造方式

* 特征图：`4 × 4`
* 序列长度：`16`
* 顺序：`row-major`

---

### 5.4 配置示例

```yaml
model:
  use_gru: true
  gru_hidden: 128
  gru_layers: 1
  bidirectional: false
  use_attention: false
  sequence_order: row
```

---

### 5.5 对比方式

| 对比模型     | 目的          |
| -------- | ----------- |
| M0 vs M1 | 验证 GRU 的有效性 |

---

### 5.6 预期结论

> GRU 能够对 token 之间的依赖关系进行建模，有效提升对复杂攻击模式的检测能力。

---

## 六、阶段三：CNN + GRU + Attention（M2 / M3）

### 6.1 研究目的

验证 Attention 在 **聚焦关键 token、抑制噪声特征** 方面的作用。

---

### 6.2 模型结构

```
Input
 → CNN
 → Tokenization
 → GRU
 → Attention Pooling
 → FC
 → Sigmoid
```

---

### 6.3 Attention 类型消融

#### M2：Additive Attention（Bahdanau）

```yaml
model:
  attn_type: add
```

#### M3：Scaled Dot-Product Attention

```yaml
model:
  attn_type: dot
```

---

### 6.4 对比方式

| 对比       | 说明                |
| -------- | ----------------- |
| M1 vs M2 | Attention 是否带来增益  |
| M2 vs M3 | 不同 Attention 机制对比 |

---

### 6.5 Attention 可解释性分析（可选加分）

* 输出 attention weights
* 绘制 token 权重分布
* 分析 DDoS 流量中高权重 token 的空间位置

---

## 七、阶段四：Token 顺序消融（结构级）

### 7.1 研究目的

验证 **不同 token 展开顺序是否影响序列建模效果**。

---

### 7.2 顺序设置

| 名称      | 说明           |
| ------- | ------------ |
| row     | 行优先          |
| z-order | Z 曲线         |
| hilbert | Hilbert-like |

```yaml
model:
  sequence_order: row / z / hilbert
```

---

### 7.3 实验对象

* 仅在 **CNN + GRU + Attention** 下进行

---

## 八、实验结果汇总表（论文模板）

| Model | GRU | Attn | Order   | F1 | ROC-AUC | PR-AUC |
| ----- | --- | ---- | ------- | -- | ------- | ------ |
| M0    | ✗   | ✗    | –       |    |         |        |
| M1    | ✓   | ✗    | row     |    |         |        |
| M2    | ✓   | Add  | row     |    |         |        |
| M3    | ✓   | Dot  | row     |    |         |        |
| M4    | ✓   | Add  | z       |    |         |        |
| M5    | ✓   | Add  | hilbert |    |         |        |

---

## 九、论文中可直接使用的消融实验总结句

> The ablation study demonstrates that introducing GRU significantly enhances the detection performance by modeling inter-feature dependencies. Furthermore, attention-based pooling enables the model to focus on discriminative tokens, leading to consistent improvements in F1-score and AUC metrics.

---

## 十、导师级建议（非常重要）

* **至少保证**：

  * M0、M1、M2 三个模型完整
* **若篇幅允许**：

  * 加入 token 顺序消融
* **Attention 可解释性**：

  * 是中文核心中极易加分的亮点

---

> ✅ **按这套方案跑完，你的实验部分是“结构清晰 + 逻辑自洽 + 可发表”的。**

如果你愿意，下一步我可以：

* 🔹 帮你写 **Ablation Study 小节的论文正文**
* 🔹 帮你生成 **对应 config 的最小可复现版本**
* 🔹 帮你设计 **图表（ROC、PR、Attention Heatmap）该怎么画**

你直接说下一步做哪一个。
