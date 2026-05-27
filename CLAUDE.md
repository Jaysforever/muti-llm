# Project: AbstainQA Reproduction (Don't Hallucinate, Abstain)

## 📌 项目背景 (Project Context)
本项目旨在复现 ACL 2024 论文《Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration》。
核心目标是实现无需微调、无需训练的多智能体协作（Multi-Agent Collaboration）框架，让大模型（LLM）学会识别知识盲区并选择"弃权（Abstain）"。

论文作者：Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, Yulia Tsvetkov (University of Washington 等)
官方代码：https://github.com/BunsenFeng/AbstainQA

---

## 🧪 论文实验全景 (Experiment Overview)

### 模型 (3个)
| 模型 | 规模 | 调用方式 |
|------|------|----------|
| Mistral-7B-Instruct | 7B | HuggingFace `mistralai/Mistral-7B-Instruct-v0.1` |
| LLaMA2-70B-Chat | 70B | HuggingFace `meta-llama/Llama-2-70b-chat-hf` |
| ChatGPT (GPT-3.5-Turbo-Instruct) | - | OpenAI API |

### 数据集 (6个)
| 数据集 | 用途 | 规模 (held-out/test) | 领域 |
|--------|------|---------------------|------|
| MMLU | 主实验 | 1000/1000 (下采样) | 57 学科通用知识 |
| Knowledge Crosswords | 主实验 | 1094/1007 (官方划分) | 多跳组合推理 |
| Hellaswag | 主实验 | 1000/1000 (随机划分) | 常识推理 |
| Propaganda | 主实验 | 231/200 (随机划分) | 宣传策略检测 |
| AmbigQA | Abstain Absolute | 1000/1000 | 歧义问题 |
| ElectionQA23 | Abstain Absolute | 67/200 (人工审核) | 时间错配 |

### 11 个 Baseline 方法（4 大类）
**Calibration-Based (基于校准):**
1. Token Probability — 用 answer token 概率作置信度，在 held-out set 上调阈值
2. Temperature Scaling — 先对 logits 做温度缩放，再调阈值
3. Ask for Calibration — 两阶段：生成答案 → 让 LLM 口述置信度概率

**Training-Based (基于训练):**
4. Hidden Layers — 提取 LLM hidden states，训练线性探针做二分类
5. External Verifier — 训练 RoBERTa-base 做外部验证器
6. Instruction Tuning — 用 `{instruction|q|answer}` 三元组微调 ChatGPT

**Prompting-Based (基于提示):**
7. Self-Reflect — 生成答案后追问 "The above answer is: A. True B. False"
8. More Information — 问 LLM "Do you need more information?" (Yes/No)
9. Generate and Match — 先不看选项生成答案，再判断是否匹配任一选项

**Consistency-Based (基于一致性):**
10. None-of-the-Above — 给选择题增加 NOTA 选项
11. Self-Consistency Threshold — 生成 5 条 CoT，计算 plurality index，在 held-out 上调阈值

### 评估指标 (4个)
| 指标 | 公式 | 含义 |
|------|------|------|
| Reliable Accuracy (R-Acc) | A/(A+C) | 在回答的问题中，有多少是正确的 |
| Effective Reliability (ER) | (A-C)/(A+B+C+D) | 正确回答比例 - 错误回答比例 |
| Abstain Accuracy (A-Acc) | (A+D)/(A+B+C+D) | 弃权决策的整体正确率 |
| Abstain F1 (A-F1) | HM(precision, recall) | 弃权决策的 F1 |

其中 A=正确回答, B=正确但弃权, C=错误回答, D=错误但弃权

---

## 🎯 核心复现策略 (Core Strategies)

### 1. COMPETE (竞争模式 / 测谎模式)
**机制**：给定问题 q 和 LLM 初次答案 a，用其他 LLM 生成 k 个错误替代答案和对应的伪造知识段落，让 LLM 在含有冲突信息的 prompt 下重新回答。如果多数情况下 LLM 改变答案 → Abstain。

**具体流程 (论文 Table 13):**
1. 获取 proposed answer：`LLM(q)` → a
2. 生成替代答案：
   - 选择题：随机选另一个未选选项
   - 开放题：用其他 LLM 生成替代答案
3. 为替代答案生成支持性知识段落：`LLM("Generate a knowledge paragraph about {alt_answer}")` 
4. 用冲突知识挑战原 LLM：`LLM("Answer the question with the following knowledge: {alt_knowledge}\nQuestion: {q}")` → new_answer
5. 比较：如果 new_answer ≠ a → 该轮判定为"动摇"
6. 默认 k=3，多数动摇则 Abstain

### 2. COOPERATE (协作模式 / 专家会诊模式)
**机制**：针对 q 和 a，生成 k 份审查反馈（Feedback），再由 Judge 汇总裁决。

**两种变体:**

**COOPERATE-Self (论文 Table 11)** — 同一 LLM 自我专精化：
1. 获取 proposed answer
2. 对 3 个领域分别生成专业知识段落并给出反馈：
   - "factual information" (事实知识)
   - "commonsense knowledge" (常识推理)  
   - "mathematical knowledge" (数学知识)
3. 对每个领域：先 prompt LLM 生成该领域相关知识 → 再让 LLM 以审稿人身份给出反馈
4. Judge (同一 LLM) 汇总所有反馈，输出 A.True / B.False
5. 若为 B.False → Abstain

**COOPERATE-Others (论文 Table 12)** — 不同 LLM 交叉审查：
1. 获取 proposed answer
2. 让其他 LLM 各自审查并给出反馈：`LLM_i("Please review the proposed answer and provide feedback on its correctness.")`
3. Judge LLM 汇总反馈，输出 A.True / B.False

---

## 🔧 所有 Prompt 模板 (From Paper Appendix B)

### Table 7: Ask for Calibration
```
Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.
For example: Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
The question is: <question>
[LLM-generated answer]
Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.
For example: Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>
Probability: [LLM-generated probability]
```

### Table 8: Self-Reflect
```
<question>
[LLM-generated answer]
The above answer is:
A. True
B. False
The answer is [LLM-generated A/B]
```

### Table 9: More Information
```
<question>
Do you need more information to answer this question? (Yes or No)
[LLM-generated yes/no]
```

### Table 10: Generate and Match
```
<question without multiple-choice options>
Proposed answer: [LLM-generated answer]
<options>
Does the proposed answer exist in the options?
[LLM-generated yes/no]
```

### Table 11: COOPERATE-Self (完整)
```
// Step 1: 获取 proposed answer
Question: <question>
Answer: [generated proposed answer]

// Step 2: 3 个领域的自我专精化审查
for domain in ["factual information", "commonsense knowledge", "mathematical knowledge"]:
    Generate some knowledge about the question, focusing on <domain>:
    [generated domain knowledge]
    Knowledge: <generated domain knowledge>
    Question: <question>
    Answer: <generated proposed answer>
    Please review the proposed answer and provide feedback on its correctness.
    Feedback: [generated feedback]

// Step 3: Judge 汇总裁决
Question: <question>
Proposed Answer: <generated proposed answer>
Feedback 1: <generated feedback from expert 1>
...
Feedback k: <generated feedback from expert k>
Based on the feedback, the proposed answer is:
A. True
B. False
The answer is [A/B].
```

### Table 12: COOPERATE-Others (完整)
```
// Step 1: 获取 proposed answer
Question: <question>
Answer: [generated proposed answer]

// Step 2: 其他 LLM 交叉审查
for llm in list_of_other_llms:
    Question: <question>
    Answer: <generated proposed answer>
    Please review the proposed answer and provide feedback on its correctness.
    Feedback: [generated feedback using llm]

// Step 3: Judge 汇总裁决
Question: <question>
Proposed Answer: <generated proposed answer>
Feedback 1: <generated feedback from llm 1>
...
Feedback k: <generated feedback from llm k>
Based on the feedback, the proposed answer is:
A. True
B. False
The answer is [A/B].
```

### Table 13: COMPETE (完整)
```
// Step 1: 获取 proposed answer
Question: <question>
Answer: [generated proposed answer]

// Step 2: 获取替代答案
if multiple-choice:
    <alternative answer> = randomly select another unchosen answer
else:
    Question: <question>
    Answer: <generated proposed answer>
    Please propose an alternative answer: [alternative answer]

// Step 3: 为替代答案生成支持段落
Question: <question>
Generate a knowledge paragraph about <alternative answer>:
[generated alternative passage]

// Step 4: 用冲突知识挑战
Answer the question with the following knowledge: feel free to ignore irrelevant or wrong information.
Knowledge: <generated alternative passage>
Question: <question>
Answer: [new generated answer]

// Step 5: 弃权判断
if <new generated answer> == <generated proposed answer>:
    abstain = False
else:
    abstain = True
```

---

## 📐 实现细节 (Implementation Details)

### 推理参数
- 默认 temperature = 0.1 (贪心解码)
- 需要多次采样时 temperature = 0.7 (如 Self-Consistency 的 5 条 CoT)

### Calibration 方法
- Token Probability: 直接用 answer token 概率，abstain likelihood = 1 - p(a)
- Temperature Scaling: 在 [0.1, 10] 范围内搜索最优 τ
- Ask for Calibration: 用 Table 7 两阶段 prompt

### Training 方法
- Hidden Layers: HuggingFace feature-extraction pipeline → 线性层 (dim, 2)
- External Verifier: RoBERTa-base 的 [CLS] token 做二分类
- Instruction Tuning: 微调 GPT-3.5-Turbo-Instruct，instruction = "Answer the following question. If you don't have enough knowledge, abstain by saying 'sorry, I don't have enough knowledge to answer this question.'"

### Consistency 方法
- Self-Consistency: k=5 条 CoT，abstain likelihood = 1 - plurality_index
- NOTA: 直接加一个 "none of the above" 选项

### 推理开销 (Table 6)
| 方法 | API 调用次数 |
|------|-------------|
| 大多数 baseline | 1-3 次 |
| COOPERATE-Self | ~8 次 |
| COOPERATE-Others | 2 + 其他 LLM 数量 |
| COMPETE | 2 + 其他 LLM 数量 |

---

## 📁 项目目录结构 (Project Structure)
```
data/              — 存放下载的数据集
results/           — 存放实验输出、日志、可视化
config.py          — 环境配置，加载 .env，定义模型名称
loader.py          — 数据集加载模块（MMLU, K-Crosswords, Hellaswag, Propaganda）
prompts.py         — 集中管理所有 Prompt 模板
compete.py         — COMPETE 策略逻辑
cooperate.py       — COOPERATE 策略逻辑（self + others）
baselines.py       — 11 个 baseline 方法
evaluator.py       — 计算 R-Acc, ER, A-Acc, A-F1
main.py            — 实验主入口
```

---

## 🚦 复现路线图 (Reproduction Roadmap)

### Phase 1: 基础设施 (预计 1-2 天)
- [ ] 1.1 创建项目骨架：所有 .py 文件、.env、requirements.txt
- [ ] 1.2 `config.py`: 加载 API Keys，定义模型名称映射，设置默认 temperature
- [ ] 1.3 `loader.py`: 实现 MMLU 数据加载（HuggingFace datasets），支持 limit 参数
- [ ] 1.4 `prompts.py`: 集中定义所有 Prompt 模板（Table 7-13）
- [ ] 1.5 验证 API 连通性（单次调用测试）

### Phase 2: 简单 Baseline (预计 2-3 天)
- [ ] 2.1 `baselines.py` — Self-Reflect (Table 8)
- [ ] 2.2 `baselines.py` — More Information (Table 9)
- [ ] 2.3 `baselines.py` — Generate and Match (Table 10)
- [ ] 2.4 `baselines.py` — None-of-the-Above
- [ ] 2.5 `baselines.py` — Token Probability
- [ ] 2.6 `baselines.py` — Ask for Calibration (Table 7)
- [ ] 2.7 `baselines.py` — Self-Consistency Threshold (k=5 CoT)
- [ ] 2.8 `evaluator.py`: 实现 4 个评估指标

### Phase 3: 核心方法 — COMPETE (预计 1-2 天)
- [ ] 3.1 `compete.py`: 单条数据完整流程
- [ ] 3.2 处理选择题 vs 开放题的替代答案生成
- [ ] 3.3 多数投票逻辑（k=3，多数动摇则 Abstain）
- [ ] 3.4 单条测试通过后，封装为批量处理函数

### Phase 4: 核心方法 — COOPERATE (预计 2-3 天)
- [ ] 4.1 `cooperate.py`: COOPERATE-Self（3 领域自我专精化）
- [ ] 4.2 `cooperate.py`: COOPERATE-Others（多模型交叉审查）
- [ ] 4.3 Judge 汇总逻辑 + JSON/A/B 解析
- [ ] 4.4 单条测试 → 批量处理

### Phase 5: 实验编排 (预计 1 天)
- [ ] 5.1 `main.py`: 命令行参数解析（--strategy, --dataset, --limit, --model）
- [ ] 5.2 批量运行 + 日志记录（loguru）
- [ ] 5.3 结果保存为 CSV/JSON，输出 4 项指标

### Phase 6: 扩展实验（可选，预计 2-3 天）
- [ ] 6.1 Abstain Absolute：AmbigQA + ElectionQA23
- [ ] 6.2 Abstain + Retrieval：WikiSearch API + 两步 abstain-retrieve-abstain
- [ ] 6.3 Multi-Hop 分析：K-Crosswords 3-hop 子集逐跳检测
- [ ] 6.4 Abstain ECE 计算
- [ ] 6.5 可视化（precision-recall 图、领域差异图）

---

## 💻 编码规范 (Coding Conventions)
1. **API 调用安全**：绝对不要在代码中硬编码 API Key。必须使用 `python-dotenv` 从 `.env` 文件中加载环境变量。
2. **容错机制**：LLM API 调用使用 `tenacity` 库添加 Retry 重试机制。
3. **结构化输出**：LLM 判断结果尽可能返回 JSON 格式，或用正则提取 "A. True" / "B. False"。
4. **日志记录**：使用 `loguru` 清晰记录模型之间的"对话日志"。
5. **小步快跑**：优先实现"单条数据的测试流"，测试通过后再写循环遍历整个数据集。
6. **虚拟环境**：在本地名称为 `nlp` 的虚拟环境上运行。
7. **函数式优先**：优先使用函数式编程，避免不必要的类继承。

## 🚀 常用指令 (Commands)
- 安装依赖: `pip install -r requirements.txt`（依赖: openai, python-dotenv, datasets, tenacity, matplotlib, loguru, scikit-learn）
- 运行单条 COMPETE 测试: `python compete.py --test`
- 运行单条 COOPERATE 测试: `python cooperate.py --test --mode self`
- 运行完整实验: `python main.py --strategy compete --dataset mmlu --limit 50 --model mistral`
- 运行所有 baseline: `python main.py --strategy all_baselines --dataset mmlu --limit 50`

## 🧠 给 Claude 的开发指南 (Instructions for Claude)
- 当我要求你"实现 XXX 模块"时，请参考上述目录结构和 Prompt 模板。
- 论文 Table 7-13 的 Prompt 模板是实现的核心依据，务必严格对齐。
- 每个 API 调用函数都需要 @retry 装饰器。
- 完成每个文件后，提醒我运行并检查 API 连通性。
- 评估指标公式务必与论文 Section 3 严格一致。
