# GPT-2 Taylor-MLP 预实验设计文档

## 1. 实验概述

在 GPT-2 Small 上验证逐层 Taylor 近似的可行性：
- 对每层 FFN 独立做 K-means + 一阶泰勒展开
- 离线测量每层的"线性度"
- 逐步替换 FFN，测量 PPL 变化
- 测量 **加速效果**：FFN 层替换前后耗时、整体模型推理耗时
- 分析多层误差累积行为

### 两阶段架构

实验分为两个独立阶段，中间通过磁盘文件衔接：

```
Phase A: 数据收集（跑一次，结果存盘）
  GPT-2 推理 → 提取 12 层 FFN IO
  → K-means 聚类 → 中心点
  → Jacobian 预计算 → F(X₀) + J(X₀)
  → 全部序列化到 Cache/GPT2/

Phase B: 验证实验（反复跑，从盘加载）
  加载 cache → 单层替换 / 累积替换 / 扫 k → 测 PPL + 计时
```

Phase B 可以反复迭代——换 k 值、换替换策略——都不需要重算 Jacobian。

---

## 2. 目录结构

```
workspace
├── GPT2/                              ← 新建实验目录
│   ├── setup.sh                       ← 环境准备脚本（安装依赖、下载数据集）
│   ├── config.py                      ← 全局配置（模型名、数据集、k 值等参数）
│   ├── eval_baseline.py               ← 基线评估：验证 GPT-2 在 OpenWebText 上的 PPL
│   ├── model.py                       ← GPT-2 模型封装 + forward hook
│   ├── collect_data.py                ← Phase A: 数据收集（特征提取 + K-means + Jacobian → 存盘）
│   ├── layer_analysis.py              ← Phase B Step 1: 逐层离线测量 (从盘加载 cache → CosSim/MSE)
│   ├── single_replace.py              ← Phase B Step 2: 单层替换测试 (加载 cache → 替换一层 → 测 PPL)
│   ├── cumulative_replace.py          ← Phase B Step 3: 累积替换测试 (加载 cache → 逐层替换 → 测 PPL)
│   ├── evaluate.py                    ← PPL / CosSim / MSE 评估工具 + 计时
│   ├── utils.py                       ← 共享工具 (K-means, Jacobian 计算, Taylor 推理, 序列化/反序列化)
│   └── run_all.py                     ← 一键运行脚本
│
├── Cache/                             ← 中间结果存储（gitignore）
│   └── GPT2/                          ← GPT-2 实验的预计算 cache
│       ├── layer_0/
│       │   ├── centers.pt             ← k 个聚类中心
│       │   ├── f_values.pt            ← k 个 F(X₀)
│       │   ├── jacobians.pt           ← k 个 J(X₀)
│       │   └── metadata.json          ← d_in, d_out, k, timestamp 等元信息
│       ├── layer_1/
│       │   └── ...
│       └── layer_11/
├── Result/
│   └── GPT2/
│       ├── step1_layer_cosim.csv      ← Phase B Step 1 结果
│       ├── step1_layer_mse.csv
│       ├── step2_single_ppl.csv       ← Phase B Step 2 结果
│       ├── step3_cumulative_ppl.csv   ← Phase B Step 3 结果
│       └── summary.csv                ← 汇总结果
│
├── docs/
│   ├── design.md                      ← 项目整体设计
│   └── gpt2.md                        ← 本文档
│
└── Workbook/
    └── workbook_004_gpt2.md           ← GPT-2 实验工作记录
```

---

## 3. 文件详解

### 3.0 `setup.sh` — 环境准备

**职责**：预下载实验所需数据。

```bash
#!/bin/bash
# 预下载数据集（缓存到本地，避免运行时下载）
python -c "from datasets import load_dataset; load_dataset('openwebtext')"

# 预下载模型（缓存到本地）
python -c "from transformers import GPT2Model, GPT2Tokenizer; GPT2Model.from_pretrained('openai-community/gpt2'); GPT2Tokenizer.from_pretrained('openai-community/gpt2')"

echo "Setup complete."
```

### 3.1 `config.py` — 全局配置

**职责**：所有可调参数集中管理。所有 Python 脚本从该文件导入配置，避免硬编码。

**核心内容**：

| 参数 | 默认值 | 说明 |
|------|------|------|
| `MODEL_NAME` | `"openai-community/gpt2"` | 模型标识，可换 GPT-2 Medium/Large |
| `DATASET_NAME` | `"openwebtext"` | OpenWebText（GPT-2 训练分布） |
| `DATASET_CONFIG` | `None` | 无需子配置 |
| `K_VALUES` | `[1, 2, 4, 8, 16, 32, 64, 128, 256]` | K-means k 扫描范围 |
| `MAX_TRAIN_SAMPLES` | `50000` | K-means 训练采样 token 数 |
| `MAX_TEST_SAMPLES` | `2000` | PPL 测试采样 token 数 |
| `DEVICE` | `"cuda" if torch.cuda.is_available() else "cpu"` | 运行设备 |
| `RESULT_DIR` | `"Result/GPT2"` | 输出目录 |
| `CACHE_DIR` | `"Cache/GPT2"` | 预计算缓存目录（gitignore） |
| `SEED` | `42` | 随机种子 |

### 3.2 `eval_baseline.py` — 基线评估

**职责**：验证 GPT-2 在 OpenWebText 上的基准 Perplexity，确认模型和数据集都正常工作。在所有 Taylor 实验之前运行。

**核心函数**：

| 符号 | 说明 |
|------|------|
| `eval_baseline(model_name, dataset_name, max_samples)` | 加载模型和数据集，跑推理，输出 PPL 和推理耗时 |

**输出**（控制台 + 文件）：

| 指标 | 说明 |
|------|------|
| `baseline_ppl` | GPT-2 原始 PPL |
| `avg_time_per_token_ms` | 每 token 平均推理耗时 |
| `total_time_s` | 总推理耗时 |
| `num_tokens` | 测试 token 数 |

**命令行**：
```bash
python eval_baseline.py                          # 用 config.py 默认参数
python eval_baseline.py --model gpt2-medium       # 换模型
python eval_baseline.py --max-samples 5000         # 自定义样本数
```

### 3.3 `model.py` — 模型封装

**职责**：加载 GPT-2 Small，注册 forward hook 拦截每层 FFN 的输入/输出。

**核心类/函数**：

| 符号 | 说明 |
|------|------|
| `GPT2Wrapper` | 封装 HuggingFace GPT2Model，注册 FFN hook |
| `register_ffn_hooks(model)` | 对每个 Transformer Block 的 MLP 子模块注册 hook |
| `get_ffn_io()` | 返回收集到的每层 FFN 输入/输出 dict |
| `clear_ffn_io()` | 清空 hook 缓存 |
| `replace_ffn_with_taylor(model, layer_idx, cache)` | 将指定层的 FFN 替换为 Taylor 近似函数 |
| `restore_ffn(model, layer_idx)` | 恢复指定层的原始 FFN |

**输入来源**：HuggingFace `openai-community/gpt2`

> 数据集用 **OpenWebText**（GPT-2 训练同分布），无需 fine-tune，直接可用预训练权重验证 Taylor 近似效果。

---

### 3.4 `collect_data.py` — Phase A: 数据收集

**职责**：跑 GPT-2 推理，提取所有层 FFN 输入/输出，做 K-means 聚类，预计算 Jacobian，全部序列化到磁盘。**跑一次即可。**

**核心函数**：

| 符号 | 说明 |
|------|------|
| `collect_ffn_io(model, tokenizer, dataset, max_samples)` | 跑推理，hook 拦截每层 FFN 的 I/O |
| `compute_and_save_layer(ffn_fn, inputs, outputs, k, save_dir)` | 对单层：K-means → 中心 → Jacobian → 存盘 |
| `collect_all_layers(model_name, dataset_name, k_values, save_root)` | 主函数：对所有层和所有 k，生成 cache |

**输出到 `Cache/GPT2/layer_{i}/`**：

```
layer_0/
├── centers.pt       ← (k, 768)  聚类中心
├── f_values.pt      ← (k, 768)  每个中心的 F(X₀)
├── jacobians.pt     ← (k, 768, 768)  Jacobian 矩阵
└── metadata.json    ← {"d_in": 768, "d_out": 768, "k": N, "timestamp": ...}
```

> 每个 k 值生成一份独立 cache，写入不同目录或 metadata 区分。

**数据流**：
```
OpenWebText → Tokenizer → GPT-2(forward) → hooks 拦截 → {layer_0: {input, output}, ...}
                                                              │
                                          ┌───────────────────┘
                                          ▼
                              K-means → centers → Jacobian → cache → 存盘
```

### 3.5 `layer_analysis.py` — Phase B Step 1: 离线测量

**职责**：从磁盘加载 cache，对每层独立做 Taylor 推理，测量 CosSim/MSE。**不替换模型，仅评估精度。**

**核心函数**：

| 符号 | 说明 |
|------|------|
| `load_cache(layer_dir)` | 从磁盘加载某层的 centers / f_values / jacobians |
| `analyze_layer_from_cache(cache, test_inputs, test_outputs)` | 基于已加载 cache 做 Taylor 推理，返回 CosSim/MSE |
| `analyze_all_layers(cache_root, features, k_values)` | 对所有 12 层做分析 |
| `plot_layer_curves(results)` | 画 12 条层曲线（k vs CosSim） |
| `identify_representative_layers(results)` | 选出最线性 / 中等 / 最非线性的 3 个代表层 |

**输出**：`step1_layer_cosim.csv`, `step1_layer_mse.csv`

| 列 | 说明 |
|------|------|
| `layer` | 层索引 (0-11) |
| `k` | K-means 聚类数 |
| `cosine_sim` | 平均余弦相似度 |
| `mse` | 均方误差 |
| `ffn_orig_time_ms` | 原始 FFN 单次前向耗时 (ms) |
| `ffn_taylor_time_ms` | Taylor 近似单次推理耗时 (ms) |
| `speedup` | Taylor / 原始 FFN 加速比 |
| `jacobian_time_s` | Jacobian 预计算耗时 (s) |
| `kmeans_time_s` | K-means 聚类耗时 (s) |

**k 值范围**：[1, 2, 4, 8, 16, 32, 64, 128, 256]

> 说明：FFN 输入为 768 维（d_model），存储平衡点 k≈96（12层总计）。k 范围拉到 256 以覆盖完整精度曲线。

---

### 3.6 `single_replace.py` — Phase B Step 2: 单层替换

**职责**：从磁盘加载 cache，替换指定的一层 FFN 为 Taylor 近似，其他层保持原样，测量 PPL 变化。

**核心函数**：

| 符号 | 说明 |
|------|------|
| `load_cache_for_layer(cache_root, layer_idx, k)` | 从磁盘加载指定层、指定 k 的 cache |
| `replace_single_layer(model, cache, layer_idx)` | 替换第 layer_idx 层 FFN 为 Taylor |
| `test_single_layer(model, tokenizer, dataset, cache_root, layer_idx, k_values)` | 对指定层测试多个 k 值 |
| `test_representative_layers(model, tokenizer, dataset, cache_root, layer_indices, k_values)` | 对多个代表层分别测试 |

**输出**：`step2_single_ppl.csv`

| 列 | 说明 |
|------|------|
| `layer` | 被替换的层索引 |
| `k` | 该层的 K-means 聚类数 |
| `ppl` | 替换后的 Perplexity |
| `delta_ppl` | PPL 增量 |
| `delta_ppl_pct` | PPL 增量百分比 |
| `cosine_sim` | 该层近似的 CosSim（来自 Step 1） |
| `model_time_orig_ms` | 原始模型单次推理耗时 (ms) |
| `model_time_taylor_ms` | 单层替换后模型推理耗时 (ms) |
| `model_speedup` | 模型整体加速比 |

**代表层选择**：基于 Step 1 结果自动选出：
- 最线性层（CosSim 最高）
- 中间层（CosSim 中位数）
- 最非线性层（CosSim 最低）

---

### 3.7 `cumulative_replace.py` — Phase B Step 3: 累积替换

**职责**：从磁盘加载 cache，从最后一层向前逐步替换，测量 PPL 随替换层数的变化。

**核心函数**：

| 符号 | 说明 |
|------|------|
| `replace_layers_backward(model, cache_root, num_layers, k)` | 从磁盘加载最后 num_layers 层的 cache，替换 FFN |
| `test_cumulative(model, tokenizer, dataset, cache_root, max_layers, k_values)` | 累积替换测试：1→2→...→12 层 |
| `test_adaptive_k(model, tokenizer, dataset, cache_root, k_map)` | 每层独立 k 值的累积替换 |

**输出**：`step3_cumulative_ppl.csv`

| 列 | 说明 |
|------|------|
| `num_replaced` | 替换层数 (1-12) |
| `k` | 每层统一 k 值 |
| `ppl` | 替换后 Perplexity |
| `delta_ppl` | PPL 增量 |
| `delta_ppl_pct` | PPL 增量百分比 |
| `storage_ratio` | Taylor 总存储 / 原始 FFN 总存储 |
| `model_time_orig_ms` | 原始模型单次推理耗时 (ms) |
| `model_time_taylor_ms` | 累积替换后模型推理耗时 (ms) |
| `model_speedup` | 模型整体加速比 |

**替换顺序**：从 Layer 11 到 Layer 0（末层向前），因为：
- 最后几层的 FFN 输出直接经过 LM head 产生 logits，影响最直接
- 从后往前替换更容易观察到误差累积的临界点

---

### 3.8 `evaluate.py` — 评估工具

**职责**：PPL 计算、CosSim/MSE 计算等通用评估函数。

**核心函数**：

| 符号 | 说明 |
|------|------|
| `compute_ppl(model, tokenizer, dataset, max_samples)` | 计算模型在数据集上的 Perplexity，同时记录整体推理耗时 |
| `compute_cosine_similarity(y_true, y_pred)` | 批量计算余弦相似度 |
| `compute_mse(y_true, y_pred)` | 批量计算 MSE |
| `baseline_ppl(model, tokenizer, dataset)` | 计算原始 GPT-2 的基线 PPL 和推理耗时 |
| `benchmark_ffn(ffn_fn, x_batch)` | 测量单次 FFN 前向的 Wall-Clock 时间 |
| `benchmark_model(model, tokenizer, dataset)` | 测量完整模型推理的 Wall-Clock 时间 |

---

### 3.9 `utils.py` — 共享工具

**职责**：K-means 聚类、Jacobian 计算、Taylor 推理、cache 序列化/反序列化。

**核心函数**：

| 符号 | 说明 |
|------|------|
| `compute_centers(data, k)` | K-means 聚类，返回 k 个中心 |
| `compute_jacobian(ffn_fn, x0)` | 在 x0 处计算 FFN 的 F(X0) 和 Jacobian J(X0) |
| `precompute_centers(ffn_fn, centers)` | 对所有中心预计算 F 和 J |
| `taylor_predict(x, cache)` | 对单个样本做 Taylor 推理：最近中心 + 一阶展开 |
| `taylor_predict_batch(x_batch, cache)` | 批量 Taylor 推理 |
| `save_cache(layer_dir, centers, f_values, jacobians, metadata)` | 将预计算结果序列化到磁盘 |
| `load_cache(layer_dir)` | 从磁盘加载预计算结果 |

---

### 3.10 `run_all.py` — 一键运行（可选）

**职责**：按顺序调用各模块，可以从头跑到尾。

**命令行接口**：
```
python run_all.py                    # 从头跑全部
python run_all.py --step 1           # 只跑 Step 1
python run_all.py --step 2           # 只跑 Step 2（需要 Step 1 结果）
python run_all.py --step 3           # 只跑 Step 3（需要 Step 1 结果）
python run_all.py --k "1,4,16,64"   # 自定义 k 值范围
python run_all.py --max-samples 20000 # 限制样本数
```

---

## 4. 数据流全景

```
Phase A (collect_data.py)
  OpenWebText → GPT-2 → hook FFN IO → K-means → Jacobian
                                                    ↓
                                          Cache/GPT2/layer_{i}/
                                          (centers.pt, f_values.pt, jacobians.pt)

Phase B (从 Cache 加载)
  Cache/GPT2/  ──→  layer_analysis.py    ──→  step1_layer_cosim.csv
                 │                                  │
                 │                    ┌──────────────┼──────────────┐
                 │                    ▼              ▼              ▼
                 │              最线性层         中等层       最非线性层
                 │                    │              │              │
                 ├────→  single_replace.py    ──→  step2_single_ppl.csv
                 │                    │
                 └────→  cumulative_replace.py ──→  step3_cumulative_ppl.csv
```

---

## 5. 实施计划

| 序号 | 任务 | 文件 | 依赖 | 说明 |
|:---:|------|------|------|------|
| 1 | 环境准备 + 配置 | `setup.sh`, `config.py` | 无 | 下载数据集和模型、定义全局参数 |
| 2 | 基线评估 | `eval_baseline.py` | 1 | 验证 GPT-2 在 OpenWebText 上的基准 PPL |
| 3 | 模型封装 + Hook | `model.py` | 1 | 加载 GPT-2，注册 FFN hook，替换/恢复 FFN |
| 4 | 共享工具 | `utils.py` | 1 | K-means、Jacobian、Taylor 推理、序列化/反序列化 |
| 5 | Phase A: 数据收集 | `collect_data.py` | 3,4 | 提取 FFN IO → K-means → Jacobian → 存盘到 Cache/GPT2/ |
| 6 | 评估工具 | `evaluate.py` | 3 | PPL、CosSim、MSE 计算 + 计时 |
| 7 | Phase B Step 1: 离线测量 | `layer_analysis.py` | 4,5,6 | 从 Cache 加载 → Taylor 推理 → CosSim/MSE |
| 8 | Phase B Step 2: 单层替换 | `single_replace.py` | 4,5,6,7 | 从 Cache 加载 → 替换一层 → 测 PPL |
| 9 | Phase B Step 3: 累积替换 | `cumulative_replace.py` | 4,5,6,7 | 从 Cache 加载 → 逐层替换 → 测 PPL |
| 10 | 一键脚本 | `run_all.py` | 1-9 | 命令行入口 |

**建议编写顺序**：1 → 2 → 3 → 5 → 6 → 4 → 7 → 8 → 9 → 10

---

## 6. 依赖清单

```txt
torch>=2.0
transformers>=4.30
datasets>=2.12
scikit-learn>=1.2
numpy
tqdm
```

---

## 7. 关键参数速查

| 参数 | 默认值 | 说明 |
|------|------|------|
| 模型 | `openai-community/gpt2` | GPT-2 Small (124M) |
| 数据集 | WikiText-2 → **OpenWebText** | `datasets.load_dataset("openwebtext")`（GPT-2 训练同分布，无需 fine-tune） |
| k 范围 | [1, 2, 4, 8, 16, 32, 64, 128, 256] | K-means 聚类数（FFN 输入 768 维，存储平衡 ~k=96） |
| 特征采样 | 50,000 tokens | K-means 所用训练样本数 |
| PPL 测试样本 | 2,000 tokens | PPL 测试所用样本数 |
| FFN 层数 | 12 | GPT-2 Small 的 Transformer Block 数 |
| d_model | 768 | 隐藏维度 |
| d_ff | 3072 | FFN 中间维度 |
| 激活函数 | GELU | 原始 GPT-2 的 FFN 激活 |

---

## 8. 预期输出概要

| 文件 | 关键列 | 预期可视化 |
|------|------|------|
| `step1_layer_cosim.csv` | layer, k, cosine_sim, speedup | 12 条曲线（k vs CosSim），每层加速比 |
| `step2_single_ppl.csv` | layer, k, delta_ppl_pct, model_speedup | 3 根柱（最线性/中等/最非线性），各自加速比 |
| `step3_cumulative_ppl.csv` | num_replaced, k, delta_ppl_pct, model_speedup | 横轴替换层数，纵轴 ΔPPL%，多 k 线 + 加速比曲线 |
| `summary.csv` | 汇总所有关键数据点（精度 + 速度） | — |

### 核心输出关系

```
k ↑  →  CosSim ↑  →  ΔPPL ↓  →  但存储 ↑  &  加速效果 ↓（最近邻搜索开销增大）
```

最终需要回答：**在满足精度要求（ΔPPL < X%）的前提下，最大模型加速比是多少？**
