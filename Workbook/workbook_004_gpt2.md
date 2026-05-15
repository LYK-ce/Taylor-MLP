# GPT-2 实验工作记录

## 任务5：Code Review GPT2/ 实现
- 开始时间: 2026-05-15

## 任务5：Code Review 结果 (完成)
- 结束时间: 2026-05-15
- 评审人: AI Agent

### 文件清单 (11个文件)
config.py, model.py, utils.py, eval_baseline.py, collect_data.py, layer_analysis.py, single_replace.py, cumulative_replace.py, evaluate.py, setup.sh, run_all.py

### 整体评价
实现基本遵循 docs/gpt2.md 设计，两阶段架构(Phase A→B)正确，函数覆盖面完整。存在一些代码质量和一致性问题。

### 发现问题汇总

#### 🔴 严重
1. **命名规范违规** — instruction.md 要求函数使用 Pascal_Snake_Case，但全部使用 lower_snake_case（PEP8）。涉及所有文件的全部函数名和方法名。

#### 🟡 中等
2. **数据准备代码重复** — eval_baseline.py, collect_data.py, single_replace.py, cumulative_replace.py 四份文件各有几乎相同的数据集加载/分块逻辑，应抽取为共享函数。
3. **compute_cosine_similarity 逐样本循环** — evaluate.py#compute_cosine_similarity 对每样本单独调 F.cosine_similarity，应使用向量化 `F.cosine_similarity(y_true, y_pred, dim=1).mean().item()`。
4. **setup.sh 预下载模型不匹配** — 下载 GPT2Model 但代码实际使用 GPT2LMHeadModel，下行时仍需额外下载。
5. **evaluate.py 死参数** — compute_ppl 的 label_key 参数从未使用。

#### 🟢 轻微
6. **Cache 目录结构与设计文档不一致** — 实现用 `layer_{idx}_k_{k}`（扁平），设计文档写 `layer_{idx}/`（嵌套含 metadata.json）。
7. **Hook 未自动清理** — register_ffn_hooks 注册的 hook 持续存在，需手动 remove_ffn_hooks，可能造成内存泄漏。
8. **layer_analysis.py 缺 pandas 依赖** — identify_representative_layers 使用 pandas 但未在依赖清单中声明。
9. **双重编码** — 数据集 tokenize 时先 encode 计数再拼接 encode 一次，应只 encode 一次。
10. **PPL 评估用非重叠 chunk** — 标准做法用 stride=1（滑动窗口），当前 stride=seq_len 浪费 token。
