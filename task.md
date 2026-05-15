# Taylor-MLP 任务清单

## 项目概览
Taylor-MLP 项目

## 任务列表

| 序号 | 任务描述 | 状态 | 评审意见 |
|------|---------|------|---------|
| 1 | 预实验：K-means + 一阶泰勒展开近似 MLP 精度验证 (ReLU) | ✅ 已完成 | CosSim仅0.73未达0.95，需改用光滑激活重试 |
| 2 | 预实验：光滑激活函数 (GELU/SiLU) 对比测试 | ✅ 已完成 | 🎉 SiLU k=1 CosSim=0.987, GELU k=1 CosSim=0.972 |
| 3 | Phase 2: MNIST CNN + Taylor-MLP (ReLU/GELU/SiLU) | ✅ 已完成 | 🎉 ReLU k=24 Acc降仅1.42%满足<2%标准！GELU/SiLU反而不如ReLU |
| 4 | Phase 4: GPT-2 Small + OpenWebText + 逐层 Taylor | 🔲 待开始 | 详见 docs/gpt2.md |
| 5 | Code Review: 结合 docs/gpt2.md 审查 GPT2/ 目录实现 | ✅ 已完成 | 发现10项问题: 1严重(命名规范) 4中等 5轻微, 详见 workbook_004_gpt2.md |

## 完成情况
- ✅ 任务1: ReLU 预实验
- ✅ 任务2: GELU/SiLU 对比——k=1 即可达 0.97+ CosSim
- ✅ 任务3: MNIST Phase 2——ReLU k=24 Acc降1.42%，真实数据验证通过
- 🔲 任务4: GPT-2 逐层 Taylor——待开始
- ✅ 任务5: Code Review GPT2/ 实现——10项问题已记录
