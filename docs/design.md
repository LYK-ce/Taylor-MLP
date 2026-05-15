# Taylor-MLP 实验设计文档

## 1. 项目背景

用 K-means 聚类 + 一阶泰勒展开近似 MLP 前向传播：

$$\hat{F}(x) = F(X_0) + J(X_0) \cdot (x - X_0)$$

将多层 MLP 的多次矩阵乘 + 激活简化为一次矩阵-向量乘（Jacobian @ dx）。

### 已完成实验

| 阶段 | 内容 | 关键结论 |
|------|------|---------|
| Phase 1 | 合成数据 + ReLU MLP (64→256→512→256→64) | ReLU CosSim 仅 0.73，未达 0.95 |
| Phase 2 | 合成数据 + 光滑激活 (GELU/SiLU) | SiLU k=1 CosSim=0.987，远超预期 |
| Phase 3 | MNIST CNN + Taylor-MLP (1568→256→64→10) | ReLU k=24 Acc降仅 1.42%，满足 <2% 标准 |

---

## 2. Phase 4: GPT-2 Small + WikiText + 逐层 Taylor

### 2.1 动机：为什么从 CNN 转向 Transformer？

| | VGG MLP（端到端） | Transformer FFN（逐层） |
|------|:---:|:---:|
| 输入维度 | 8192 | 768 |
| 输出维度 | 10~100 | 768 |
| Jacobian 形状 | 矩形 (d_out ≪ d_in) | **方阵 (d_in = d_out)** |
| 单中心 Jacobian | 0.08M~0.82M elem | 0.59M elem |
| 存储平衡 k | 15~550 | **4000+** |

Transformer FFN 的 $d_{in}=d_{out}=d_{model}$ 使 Jacobian 是方阵，存储极度宽松，是 Taylor-MLP 最理想的目标。此外：

- FFN 占 Transformer 总参数的 ~2/3，近似收益最大
- GPT 架构是当前主流，实验结论有实际价值
- 多层结构天然支持层间差异的消融分析

### 2.2 模型与数据

| 项目 | 选择 |
|------|------|
| 模型 | GPT-2 Small（12 层，d_model=768，d_ff=3072） |
| 预训练权重 | `openai-community/gpt2`（HuggingFace） |
| 评估数据集 | WikiText-2 |
| FFN 结构 | Linear(768→3072) → GELU → Linear(3072→768) |
| 单层 FFN 参数 | 768×3072 + 3072×768 ≈ **4.7M**（不含 bias） |
| 所有 FFN 总参数 | ~56M（占 GPT-2 Small 的 ~67%） |

### 2.3 Taylor 参数计算

单中心 Jacobian：$d_{out} \times d_{in} = 768 \times 768 = 589,824$ 元素 = **2.36 MB** (fp32)

完整存储：$k \times (J + F(X_0) + X_0) = k \times (589824 + 768 + 768) \approx k \times 591,360$ 参数

| k | Taylor 总参数 | 存储比 (vs 4.7M/层) | 存储比 (vs 56M/12层) |
|:---:|:---:|:---:|:---:|
| 1 | 0.59M | 12.5% | 1.0% |
| 4 | 2.37M | 50.1% | 4.2% |
| 8 | 4.73M | 100.1% | 8.4% |
| 16 | 9.46M | 200% | 16.7% |
| 32 | 18.9M | 400% | 33.5% |
| 64 | 37.8M | 800% | 67.0% |

**注意**：k ≤ 7 时存储小于单层 FFN；k ≤ 95 时总存储小于全 12 层 FFN。如果每层独立的 k 不同（某些层用更少中心），总存储可进一步优化。

### 2.4 核心挑战：多层误差累积

GPT 每层结构：

```
Layer i:  x → Attention(x) → x + Attn(x) → FFN(x + Attn(x)) → residual → Layer i+1
```

如果 FFN 被 Taylor 近似 $\hat{F}$ 替换，误差 $\epsilon_i = \|F(x) - \hat{F}(x)\|$ 通过残差流进入下一层。替换层数越多，误差越严重。

**误差来源**：
1. 每层 Taylor 近似的固有误差
2. 上层近似误差导致本层输入分布偏移（K-means 中心不再匹配）
3. 残差连接将误差注入两条路径（residual path + FFN path）

### 2.5 实验设计：三阶段递进

#### Step 1 — 离线测量（不替换模型，仅观察）

- 跑 GPT-2 Small 在 WikiText-2 上前向推理
- 用 forward hook 收集每层 FFN 的输入/输出 hidden states
- 对每层独立做 K-means + Taylor，测量 CosSim / MSE

**输出**：12 条曲线（每层 k vs CosSim），回答：
- 哪层最"线性"（最易近似）？
- 前几层 vs 后几层的线性度差异？
- 不同 k 下各层 CosSim 的分布？

#### Step 2 — 单层替换（隔离测试）

- 基于 Step 1 选出 3 个代表性层：最线性 / 中等 / 最非线性
- 每轮只替换其中**一层**的 FFN 为 Taylor，其他层保持原始
- 在不同 k 下测试 Perplexity 变化

**输出**：替换第 i 层 vs 第 j 层的 PPL 增量差异，回答：
- 不同位置的层对近似误差的敏感度？
- 单层替换的 PPL 损失和 CosSim 的关系？

#### Step 3 — 累积替换（逐步从末层向前）

- 从最后一层（第 12 层）开始向前逐步替换
- 每步多替换一层，记录 PPL 变化
- 扫描不同的 k 值组合（全层统一 k vs 每层独立 k）

**输出**：替换层数 vs PPL 曲线，回答：
- 能替换多少层而 PPL 仍在可接受范围？
- 误差累积是线性还是超线性？
- 最优的层分配策略（哪些层用更多中心、哪些用更少）？

#### Step 4（可选） — 激活函数对比

- 将 GPT-2 的 GELU 换为 ReLU/SiLU，重新记录 Step 1-3 的数据
- 对比三种激活在 Transformer FFN 上的逐层线性度差异

### 2.6 K-means 的特殊考虑

GPT 的自回归特性：K-means 对象是所有 token 位置的 hidden states。

```
"the cat sat on the mat" → 6 tokens × 768 dim → 6 个 K-means 样本
```

不同位置 token 的 hidden states 分布可能差异显著（开头 vs 中间 vs 末尾）。初始实验统一聚类，后续可尝试按位置分组聚类。

**K-means 采样方案**：
- WikiText-2 训练集约 2M tokens
- 采样 50K-100K token 的 hidden states 做 K-means
- 用 MiniBatchKMeans 处理高 k 值

### 2.7 评估指标

| 指标 | 含义 |
|------|------|
| Perplexity (PPL) | GPT-2 在 WikiText-2 上的语言模型困惑度，越低越好 |
| Δ PPL | Taylor 替换后的 PPL 增量 |
| Cosine Similarity | 每层 FFN 输出向量 vs Taylor 近似输出的余弦相似度 |
| MSE | 每层 FFN 输出向量 vs Taylor 近似输出的均方误差 |
| 存储比 | Taylor 参数总量 / 原始 FFN 参数量（单层或总计） |

### 2.8 成功标准

- 离线测量：存在多数层在 k ≤ 4 时 CosSim > 0.95
- 单层替换：替换任一层 FFN 导致的 Δ PPL < 5%
- 累积替换：存在某种层分配方案使 Δ PPL < 10% 且总存储 ≤ 全部原始 FFN
- 层间差异分析：明确哪些层对 Taylor 近似更敏感

---

## 3. 目录结构

```
workspace
├── Pre_Experiment/
│   ├── main.py                    ← Phase 1: ReLU 合成数据
│   ├── smooth_activation.py       ← Phase 2: GELU/SiLU 合成数据
│   ├── mnist_cnn.py               ← Phase 3: MNIST CNN
│   └── gpt2_taylor.py             ← Phase 4: GPT-2 + 逐层 Taylor (待实现)
├── Result/
│   └── Pre_Experiment/
├── docs/
│   ├── pre_experiment.md          ← Phase 1-3 详细设计
│   ├── design.md                  ← 本文档：Phase 4 设计
│   └── taylor_mlp.md
├── Workbook/
│   ├── workbook_001_pre_experiment.md
│   ├── workbook_002_smooth_activation.md
│   └── workbook_003_mnist_cnn.md
├── task.md
└── readme.md
```
