"""
prompts.py — 论文 Appendix B 中 Table 7~13 的所有 Prompt 模板。

每个模板使用 Python str.format() 风格的占位符 {placeholder}，
调用时用 .format(placeholder="value") 或 .format_map(dict) 填充。
"""

# ===========================================================================
# Table 7: Ask for Calibration
# ===========================================================================
ASK_CALIBRATION_GUESS = """\
Provide your best guess for the following question. Give ONLY the guess, no other words or explanation.
For example: Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
The question is: {question}"""

ASK_CALIBRATION_PROBABILITY = """\
{question}
{generated_answer}
Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.
For example: Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>
Probability:"""


# ===========================================================================
# Table 8: Self-Reflect
# ===========================================================================
SELF_REFLECT = """\
{question}
{generated_answer}
The above answer is:
A. True
B. False
The answer is"""


# ===========================================================================
# Table 9: More Information
# ===========================================================================
MORE_INFORMATION = """\
{question}
Do you need more information to answer this question? (Yes or No)"""


# ===========================================================================
# Table 10: Generate and Match
# ===========================================================================
GENERATE_ANSWER = """\
{question_without_options}
Proposed answer:"""

GENERATE_AND_MATCH = """\
{question_without_options}
Proposed answer: {generated_answer}
{options}
Does the proposed answer exist in the options?"""


# ===========================================================================
# Table 11: COOPERATE-Self
# ===========================================================================
COOPERATE_SELF_PROPOSED = """\
Question: {question}
Answer:"""

COOPERATE_SELF_KNOWLEDGE = """\
Generate some knowledge about the question, focusing on {domain}:"""

COOPERATE_SELF_REVIEW = """\
Knowledge: {domain_knowledge}
Question: {question}
Answer: {generated_answer}
Please review the proposed answer and provide feedback on its correctness.
Feedback:"""

COOPERATE_SELF_JUDGE = """\
Question: {question}
Proposed Answer: {generated_answer}
Feedback 1: {feedback_1}
Feedback 2: {feedback_2}
Feedback 3: {feedback_3}
Based on the feedback, the proposed answer is:
A. True
B. False
The answer is"""

COOPERATE_SELF_DOMAINS = [
    "factual information",
    "commonsense knowledge",
    "mathematical knowledge",
]


# ===========================================================================
# Table 12: COOPERATE-Others
# ===========================================================================
COOPERATE_OTHERS_PROPOSED = """\
Question: {question}
Answer:"""

COOPERATE_OTHERS_REVIEW = """\
Question: {question}
Answer: {generated_answer}
Please review the proposed answer and provide feedback on its correctness.
Feedback:"""

COOPERATE_OTHERS_JUDGE = """\
Question: {question}
Proposed Answer: {generated_answer}
Feedback 1: {feedback_1}
Feedback 2: {feedback_2}
Feedback 3: {feedback_3}
Based on the feedback, the proposed answer is:
A. True
B. False
The answer is"""


# ===========================================================================
# Table 13: COMPETE
# ===========================================================================
COMPETE_PROPOSED = """\
Question: {question}
Answer:"""

COMPETE_ALTERNATIVE_ANSWER = """\
Question: {question}
Answer: {generated_answer}
Please propose an alternative answer:"""

COMPETE_KNOWLEDGE = """\
Question: {question}
Generate a knowledge paragraph about {alternative_answer}:"""

COMPETE_CHALLENGE = """\
Answer the question with the following knowledge: feel free to ignore irrelevant or wrong information.
Knowledge: {alternative_passage}
Question: {question}
Answer:"""


# ===========================================================================
# 通用 QA Prompt
# ===========================================================================
QA_STANDARD = """\
Question: {question}
{options}
Answer:"""

QA_COT = """\
Question: {question}
{options}
Let's think step by step.
Answer:"""

# ===========================================================================
# De-reasoning: 强制模型只输出选项字母，禁用推理
# 用于击碎模型对未知事件的"虚假自信"
# ===========================================================================
NO_REASONING_PREFIX = (
    "Respond with ONLY the single letter of your choice (A, B, C, or D). "
    "Absolutely no explanation, no reasoning, no punctuation. "
    "Just the letter.\n\n"
)

COMPETE_PROPOSED_NO_REASONING = (
    NO_REASONING_PREFIX +
    "Question: {question}\nAnswer:"
)

COMPETE_CHALLENGE_NO_REASONING = (
    NO_REASONING_PREFIX +
    "Answer the question with the following knowledge. "
    "Respond with ONLY the letter.\n"
    "Knowledge: {alternative_passage}\n"
    "Question: {question}\n"
    "Answer:"
)
