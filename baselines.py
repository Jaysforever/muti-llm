"""
baselines.py — 7 个 AbstainQA Baseline 方法（增强版）。

改进:
  - _strip_thinking_process: 更健壮的 <think> 标签去除
  - TokenProb: 真实 logprobs 提取 + fallback
  - Gen+Match: 提取干净答案后再匹配选项
  - MoreInfo: thinking 块剥离后再解析
"""
import re
import math
from collections import Counter

from loguru import logger

from config import call_llm, call_llm_with_logprobs, DEFAULT_TEMPERATURE
from prompts import (
    SELF_REFLECT,
    MORE_INFORMATION,
    GENERATE_AND_MATCH,
    ASK_CALIBRATION_GUESS,
    ASK_CALIBRATION_PROBABILITY,
)


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _strip_thinking_process(text: str) -> str:
    """
    彻底移除 thinking 模型的推理痕迹。

    处理:
      - <think>...</think> 块（含跨行）
      - 残余的 <think> 开标签（未闭合）
      - 残余的 </think> 闭标签
      - 开头空行

    返回干净的纯文本。
    """
    # 1. 移除完整 <think>...</think> 块（支持跨行、嵌套不敏感）
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # 2. 移除未闭合的 <think> 标签及其后所有内容
    text = re.sub(r"<think>[\s\S]*", "", text)
    # 3. 移除孤立的闭标签
    text = re.sub(r"</think>", "", text)
    # 4. 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_mc_question(question: str, options: list[str]) -> str:
    """格式化多选题 prompt。纯 A/B/C/D 四选一格式。"""
    opts = "\n".join(options)
    return f"Question: {question}\n{opts}\n\nChoose one answer from the above choices.\nAnswer:"


def _parse_choice(text: str, default: str = "") -> str:
    """
    从 LLM 输出中提取选项标签 A/B/C/D/NOTA。
    先剥离 thinking 块，再在剩余纯文本中匹配。
    """
    text = _strip_thinking_process(text)
    # 模式1: 行首即是 A/B/C/D
    m = re.match(r"\s*([A-D])\s*$", text.split("\n")[0].strip())
    if m:
        return m.group(1)
    m = re.match(r"^([A-D])\b", text)
    if m:
        return m.group(1)
    # 模式2: 寻找行末或独立行出现的选项标记
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^([A-D])[\.\)\s:]", line)
        if m:
            return m.group(1)
        m = re.match(r"^[选项]*\s*([A-D])$", line)
        if m:
            return m.group(1)
    # 模式3: 最后出现的 "A." / "B." / etc
    m = re.findall(r"\b([A-D])[\.\)\s]", text)
    if m:
        return m[-1]
    # 模式4: "answer is A" / "答案是 A" / "选 A"
    m = re.search(r"(?:answer|答案|选)\s*(?:is|是|择)?\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # 模式5: NOTA / None of the above
    if re.search(r"none\s*of\s*the\s*above|NOTA", text, re.IGNORECASE):
        return "NOTA"
    return default


def _parse_true_false(text: str) -> bool | None:
    """解析 True/False 或 A/B 输出，True=回答正确。"""
    text = _strip_thinking_process(text)
    t = text.strip().upper()
    if t.startswith("A") or "TRUE" in t:
        return True
    if t.startswith("B") or "FALSE" in t:
        return False
    m = re.search(r"(A\.?\s*True|B\.?\s*False|True|False)", text, re.IGNORECASE)
    if m:
        return "true" in m.group(1).lower()
    return None


def _parse_yes_no(text: str) -> bool | None:
    """解析 Yes/No 输出，True=Yes。"""
    text = _strip_thinking_process(text)
    t = text.strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return None


def _parse_probability(text: str) -> float | None:
    """从文本中提取 0.0-1.0 之间的概率值。"""
    text = _strip_thinking_process(text)
    nums = re.findall(r"(\d+\.?\d*)", text)
    for n in nums:
        val = float(n)
        if 0.0 <= val <= 1.0:
            return val
    pcts = re.findall(r"(\d+)%", text)
    for p in pcts:
        val = float(p) / 100.0
        if 0.0 <= val <= 1.0:
            return val
    return None


def _extract_clean_answer(text: str) -> str:
    """
    从 thinking 模型输出中提取干净的答案文本。

    策略: 剥离 <think> 块后，取第一个非空段落作为答案。
    如果仍包含过长内容，进一步截取。
    """
    clean = _strip_thinking_process(text)
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    if not lines:
        return clean[:200]
    # 跳过以 "Question:" / "Let me" / "We need" 等开头的元文本行
    for line in lines:
        if re.match(r"^(Question:|Let\s|We\s|I\s|The\suser|First|Here|In\sthis)", line, re.IGNORECASE):
            continue
        return line[:200]
    return lines[0][:200] if lines else clean[:200]


# ═══════════════════════════════════════════════════════════════
# Baseline 1: Self-Reflect (Table 8)
# ═══════════════════════════════════════════════════════════════

def baseline_self_reflect(
    question: str,
    options: list[str],
    model_name: str | None = None,
) -> dict:
    mc_prompt = _format_mc_question(question, options)
    answer_raw = call_llm(mc_prompt, model_name=model_name)
    parsed = _parse_choice(answer_raw)

    reflect_prompt = SELF_REFLECT.format(
        question=mc_prompt,
        generated_answer=answer_raw,
    )
    judge_raw = call_llm(reflect_prompt, model_name=model_name)
    is_true = _parse_true_false(judge_raw)
    abstain = (is_true is not None and not is_true)

    detail = f"answer={parsed}, correct={is_true}"
    logger.info(f"[Self-Reflect] abstain={abstain} | {detail}")
    return {"abstain": abstain, "answer": answer_raw, "parsed_answer": parsed, "detail": detail}


# ═══════════════════════════════════════════════════════════════
# Baseline 2: More Information (Table 9) — 优化版
# ═══════════════════════════════════════════════════════════════

def baseline_more_info(
    question: str,
    options: list[str],
    model_name: str | None = None,
) -> dict:
    mc_prompt = _format_mc_question(question, options)
    prompt = MORE_INFORMATION.format(question=mc_prompt)
    raw = call_llm(prompt, model_name=model_name)
    needs_more = _parse_yes_no(raw)
    abstain = (needs_more is True)

    # 同时提取答案用于后续评估
    parsed = _parse_choice(raw)

    detail = f"needs_more={needs_more}, parsed={parsed}"
    logger.info(f"[MoreInfo] abstain={abstain} | {detail}")
    return {"abstain": abstain, "answer": raw, "parsed_answer": parsed, "detail": detail}


# ═══════════════════════════════════════════════════════════════
# Baseline 3: Generate and Match (Table 10) — 优化版
# ═══════════════════════════════════════════════════════════════

def baseline_gen_match(
    question: str,
    options: list[str],
    model_name: str | None = None,
) -> dict:
    # Step 1: 不看选项生成答案
    gen_prompt = f"Question: {question}\nProposed answer:"
    gen_raw = call_llm(gen_prompt, model_name=model_name)
    clean_answer = _extract_clean_answer(gen_raw)
    parsed = _parse_choice(gen_raw)

    # Step 2: 用干净答案去匹配选项
    opts = "\n".join(options)
    match_prompt = GENERATE_AND_MATCH.format(
        question_without_options=question,
        generated_answer=clean_answer,
        options=opts,
    )
    match_raw = call_llm(match_prompt, model_name=model_name)
    exists = _parse_yes_no(match_raw)
    abstain = (exists is False)

    detail = f"clean={clean_answer[:60]}, parsed={parsed}, exists={exists}"
    logger.info(f"[Gen+Match] abstain={abstain} | {detail}")
    return {"abstain": abstain, "answer": gen_raw, "parsed_answer": parsed, "detail": detail}


# ═══════════════════════════════════════════════════════════════
# Baseline 4: None-of-the-Above
# ═══════════════════════════════════════════════════════════════

def baseline_nota(
    question: str,
    options: list[str],
    model_name: str | None = None,
) -> dict:
    extended = list(options) + ["E: None of the above"]
    mc_prompt = _format_mc_question(question, extended)
    raw = call_llm(mc_prompt, model_name=model_name)
    parsed = _parse_choice(raw)
    abstain = (parsed == "NOTA" or parsed == "E")

    detail = f"parsed={parsed}"
    logger.info(f"[NOTA] abstain={abstain} | {detail}")
    return {"abstain": abstain, "answer": raw, "parsed_answer": parsed, "detail": detail}


# ═══════════════════════════════════════════════════════════════
# Baseline 5: Token Probability — 增强版（真实 logprobs + fallback）
# ═══════════════════════════════════════════════════════════════

def baseline_token_prob(
    question: str,
    options: list[str],
    model_name: str | None = None,
    threshold: float = 0.5,
) -> dict:
    mc_prompt = _format_mc_question(question, options)

    # 尝试获取 logprobs，失败则回退到普通调用
    logprobs = None
    content = ""
    try:
        content, logprobs = call_llm_with_logprobs(
            mc_prompt, model_name=model_name, max_tokens=32
        )
    except Exception as e:
        logger.warning(f"[TokenProb] logprobs call failed ({e}), falling back to regular call")
        try:
            content = call_llm(mc_prompt, model_name=model_name)
        except Exception as e2:
            logger.error(f"[TokenProb] regular call also failed: {e2}")
            return {"abstain": True, "answer": "", "parsed_answer": "", "detail": f"error: {e2}"}

    parsed = _parse_choice(content)
    prob = 1.0
    prob_source = "default"

    if logprobs and logprobs.get("top_logprobs"):
        prob_source = "logprobs"
        found_match = False
        for token_idx, top_dict in enumerate(logprobs["top_logprobs"]):
            if not top_dict:
                continue
            best_prob = 0.0
            best_token = ""
            for token_str, logp in top_dict.items():
                token_clean = token_str.strip().upper()
                p = math.exp(logp)
                if token_clean in ("A", "B", "C", "D") and p > best_prob:
                    best_prob = p
                    best_token = token_clean
            if best_token:
                found_match = True
                if best_token == parsed:
                    prob = best_prob
                    break
                # 继续看后面 token
        if found_match and prob == 1.0 and parsed:
            # 模型选了某个选项但该选项不在 top logprobs 中 → 低置信度
            prob = 0.0
    else:
        prob_source = "heuristic"
        if parsed and parsed in ("A", "B", "C", "D"):
            prob = 0.6
        elif not parsed:
            prob = 0.0

    abstain_likelihood = 1.0 - prob
    abstain = abstain_likelihood > threshold

    detail = (
        f"parsed={parsed}, prob={prob:.4f} (src={prob_source}), "
        f"abstain_likelihood={abstain_likelihood:.4f}"
    )
    logger.info(f"[TokenProb] abstain={abstain} | {detail}")
    return {"abstain": abstain, "answer": content, "parsed_answer": parsed, "detail": detail}


# ═══════════════════════════════════════════════════════════════
# Baseline 6: Ask for Calibration (Table 7)
# ═══════════════════════════════════════════════════════════════

def baseline_ask_calibrate(
    question: str,
    options: list[str],
    model_name: str | None = None,
    threshold: float = 0.5,
) -> dict:
    mc_prompt = _format_mc_question(question, options)

    guess_prompt = ASK_CALIBRATION_GUESS.format(question=mc_prompt)
    guess_raw = call_llm(guess_prompt, model_name=model_name)
    parsed = _parse_choice(guess_raw)

    prob_prompt = ASK_CALIBRATION_PROBABILITY.format(
        question=mc_prompt,
        generated_answer=guess_raw,
    )
    prob_raw = call_llm(prob_prompt, model_name=model_name)
    prob = _parse_probability(prob_raw)

    if prob is None:
        prob = 0.5
    abstain = (prob < threshold)

    detail = f"parsed={parsed}, verbalized_prob={prob:.4f}"
    logger.info(f"[AskCali] abstain={abstain} | {detail}")
    return {"abstain": abstain, "answer": guess_raw, "parsed_answer": parsed, "detail": detail}


# ═══════════════════════════════════════════════════════════════
# Baseline 7: Self-Consistency Threshold
# ═══════════════════════════════════════════════════════════════

def baseline_sc_threshold(
    question: str,
    options: list[str],
    model_name: str | None = None,
    k: int = 5,
    threshold: float = 0.5,
) -> dict:
    opts_text = "\n".join(options)
    cot_prompt = (
        f"Question: {question}\n"
        f"{opts_text}\n\n"
        f"Let's think step by step.\n"
        f"Answer:"
    )

    answers = []
    for i in range(k):
        raw = call_llm(cot_prompt, model_name=model_name, temperature=0.7)
        parsed = _parse_choice(raw)
        answers.append(parsed)
        logger.debug(f"[SC-Thres] run {i+1}/{k}: parsed={parsed}")

    if not answers or all(a == "" for a in answers):
        return {"abstain": True, "answer": "", "parsed_answer": "", "detail": "all empty"}

    counts = Counter(a for a in answers if a)
    max_count = counts.most_common(1)[0][1] if counts else 0
    plurality_index = max_count / k
    abstain = plurality_index < threshold

    detail = f"answers={answers}, plurality={plurality_index:.2f}, max_count={max_count}/{k}"
    logger.info(f"[SC-Thres] abstain={abstain} | {detail}")
    majority_answer = counts.most_common(1)[0][0] if counts else ""
    return {"abstain": abstain, "answer": str(answers), "parsed_answer": majority_answer, "detail": detail}


# ═══════════════════════════════════════════════════════════════
# 注册表
# ═══════════════════════════════════════════════════════════════

BASELINE_REGISTRY = {
    "self_reflect": baseline_self_reflect,
    "more_info": baseline_more_info,
    "gen_match": baseline_gen_match,
    "nota": baseline_nota,
    "token_prob": baseline_token_prob,
    "ask_calibrate": baseline_ask_calibrate,
    "sc_threshold": baseline_sc_threshold,
}
