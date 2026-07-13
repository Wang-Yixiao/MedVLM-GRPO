# VQA-RAD Smoke SFT 完整评估报告

## 1. 评估对象

- 基础模型：`models/Qwen2.5-VL-3B-Instruct`
- Adapter：`output/vqarad-smoke/bridge_sft_lora`
- 训练性质：Smoke SFT，仅使用 VQA-RAD 训练集前 100 条样本
- 强化学习状态：未执行策略梯度更新；已有 GRPO rollout 仅用于奖励诊断
- 测试集：VQA-RAD 官方 test split，共 451 条问答、203 张唯一图像
- 随机种子：42

因此，本报告评价的是现有 Smoke SFT adapter，而不是最终 GRPO/DAPO 模型。

## 2. 评估配置

- 推理精度：FP16
- Batch size：2
- 最大生成长度：128 tokens
- 解码方式：Greedy decoding（`do_sample=False`）
- Padding：Left padding
- 输出格式：`<think>...</think><answer>...</answer>`
- 答案评估：仅抽取 `<answer>` 内容，统一大小写与首尾空白

由于当前 PyTorch 使用 CUDA 13.0，而已安装的 bitsandbytes 不包含 CUDA 13.0 预编译库，本次关闭 4-bit 量化，使用 FP16 完成推理。该调整只改变模型加载精度，不改变 adapter 权重。

## 3. 数据完整性

| 检查项 | 结果 |
|---|---:|
| 测试问答数 | 451 |
| 唯一预测 ID | 451 |
| 空预测 | 0 |
| Closed-ended | 316 |
| Open-ended | 135 |
| 格式完全合规 | 449 / 451（99.56%） |

Closed/Open 使用确定性规则分类：yes/no 标准答案，或以 `is/are/does/how many/which side` 等封闭式问法开头的样本归为 closed，其余归为 open。该分类是工程规则，不等同于数据集人工题型标注。

## 4. 总体结果

| 指标 | 结果 |
|---|---:|
| Exact Match | 45.45% |
| ROUGE-L | 48.86% |
| METEOR | 25.16% |
| BERTScore Precision | 97.70% |
| BERTScore Recall | 97.72% |
| BERTScore F1 | 97.68% |
| 临床矛盾率 | 21.51% |
| 极性矛盾数 | 90 |
| 左右侧矛盾数 | 7 |

BERTScore 使用本地 `stsb-roberta-base` 的第 12 层计算。其绝对值明显偏高，与 Exact Match、ROUGE-L 和 METEOR 的趋势不一致，说明该模型/层配置对医学短答案的区分度有限。BERTScore 更适合在相同配置下比较不同实验组，不应单独作为正确性结论。

## 5. 分题型结果

| 题型 | 数量 | Exact Match | ROUGE-L | 临床矛盾率 |
|---|---:|---:|---:|---:|
| Closed-ended | 316 | 57.28% | 58.62% | 29.75% |
| Open-ended | 135 | 17.78% | 25.67% | 2.22% |

补充统计：

- Yes/No 子集：251 条，Exact Match 为 63.35%；
- 含 left/right 标准答案的样本：46 条，Exact Match 为 19.57%；
- 生成平均长度：6.28 words；
- 生成中位长度：6 words；
- 最大生成长度：13 words。

模型在 closed-ended 问题上明显优于 open-ended 问题，但 closed-ended 临床矛盾率达到 29.75%，说明较高的闭合题准确率并不代表模型具备可靠的极性和位置判断能力。

## 6. 输出格式与推理质量

451 条输出中，449 条严格满足 `<think>...</think><answer>...</answer>` 格式，格式合规率为 99.56%。两条异常分别为：

- ID 209：仅输出 `bowel`，缺少结构标签；
- ID 245：输出 `<answer no></answer>`，标签语法错误。

但是，450/451（99.78%）的输出使用完全相同的推理模板：

```text
Review the relevant visible medical finding.
```

这说明模型主要学会了格式，而没有学会针对图像生成实例级临床依据。当前 `<think>` 不应被解释为有效医学推理，也不能证明模型具备可解释性。

## 7. 错误分析

### 7.1 极性错误

共检测到 90 个 yes/no 或 present/absent 冲突。例如：

- 问题：`Are the kidneys present in this image?`
- 预测：`yes`
- 标准答案：`no`

极性冲突是当前最主要的临床安全问题，建议在后续训练中提高 polarity reward 权重，并分别报告 yes/no sensitivity、specificity 和 confusion matrix。

### 7.2 左右侧错误

共检测到 7 个明确的 left/right 冲突；含左右侧答案样本的 Exact Match 仅为 19.57%。例如：

- 问题：`Where is the colon most prominent from this view?`
- 预测：`right side`
- 标准答案：`left`

建议增加左右侧专项样本、图像方向增强和 laterality hard-negative reward。

### 7.3 开放式回答能力不足

Open-ended Exact Match 为 17.78%，ROUGE-L 为 25.67%。典型错误包括：

- 将 `the right bronchus` 预测为 `nutcracker sign`；
- 将 `skull, cartilage and medulla` 预测为 `bones and soft tissues`；
- 将不可见器官问题预测为影像位置词。

这些错误表明模型对解剖结构、异常描述及复杂位置关系的学习不足。

### 7.4 答案分布偏置

最高频预测包括：

| 答案 | 次数 |
|---|---:|
| yes | 178 |
| no | 72 |
| right | 13 |
| axial | 13 |
| vasculature | 5 |
| MRI | 5 |

`yes` 占全部预测的 39.47%，显示模型存在明显的高频答案偏置。这与训练数据中的 yes/no 高频分布有关，也会放大极性错误。

## 8. 有效性限制

1. **不是最终强化学习模型**：adapter 仅完成 Smoke SFT，GRPO rollout 没有更新策略参数。
2. **训练规模很小**：仅使用训练集前 100 条数据，无法代表完整训练效果。
3. **官方划分存在图像泄漏**：数据审计发现 VQA-RAD 官方 train/test 之间有 202 张重复图像。虽然评估加载器启用了 `strict_image_split`，但现有 adapter 在训练时使用的是 `strict_image_split=False`，所以本次结果仍可能受到图像重叠影响。
4. **只有一个 seed**：当前只有 seed 42 的训练产物，无法计算跨 seed 均值和标准差。
5. **没有基线对照**：尚未在完全相同的测试配置下评估 base、完整 SFT、GRPO 和 DAPO 模型，不能归因训练方法带来的提升。
6. **题型分类为规则分类**：closed/open 结果可能与人工标注存在差异。
7. **BERTScore 校准有限**：本地 STS 模型在医学短答案上给出过高分数，应以 EM、临床矛盾率和人工核验为主。

## 9. 结论

现有 Smoke SFT adapter 已稳定掌握结构化输出格式，整体 Exact Match 为 45.45%，在 closed-ended 问题上达到 57.28%。但模型几乎全部使用固定推理模板，open-ended Exact Match 仅为 17.78%，临床矛盾率达到 21.51%，左右侧子集表现尤其较弱。

因此，该模型可以证明训练、批量推理和评估链路已经打通，但尚不能证明模型获得了可靠的医学视觉推理能力，也不能作为 GRPO/DAPO 的最终实验结果。

## 10. 后续正式实验要求

- 在严格图像去重后重新训练所有模型；
- 使用相同数据、seed 和生成配置评估 Base、SFT、GRPO、DAPO；
- 至少运行 3 个 seed，报告 mean ± sample standard deviation；
- 增加 yes/no confusion matrix、laterality accuracy、BERTScore、METEOR 和格式合规率；
- 对固定推理模板、幻觉、否定和左右侧错误进行人工抽样；
- 将官方划分与 Strict Split 结果分表报告。

## 11. 评估产物

- 逐样本预测：`reports/vqarad-smoke-sft-seed42-predictions.jsonl`
- 机器可读指标：`reports/vqarad-smoke-sft-seed42-metrics.json`
- 本评估报告：`reports/vqarad-smoke-sft-seed42-evaluation.md`
