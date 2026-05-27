"""
cooperate.py — COOPERATE 策略（协作模式 / 专家会诊模式）。

支持两种模式:
  - COOPERATE-Self:  同一模型扮演 3 个领域专家 (factual/commonsense/math)
  - COOPERATE-Others: 不同模型交叉审查

用法:
  python cooperate.py --test              # 单条调试
  python cooperate.py --limit 3 --mode self    # Self 模式
  python cooperate.py --limit 3 --mode others   # Others 模式
  python cooperate.py --limit 3 --both           # 两种模式对比
"""
import sys
import re
import argparse

from loguru import logger

from config import call_llm, get_other_models, DEFAULT_MODEL
from baselines import _strip_thinking_process, _parse_choice, _format_mc_question
from prompts import (
    COOPERATE_SELF_KNOWLEDGE,
    COOPERATE_SELF_REVIEW,
    COOPERATE_SELF_JUDGE,
    COOPERATE_SELF_DOMAINS,
    COOPERATE_OTHERS_REVIEW,
    COOPERATE_OTHERS_JUDGE,
)


# ═══════════════════════════════════════════════════════════════════════
# Judge 投票解析
# ═══════════════════════════════════════════════════════════════════════

def judge_vote(judge_raw: str) -> bool | None:
    """
    从 Judge 输出中提取 True/False 决策。

    Returns
    -------
    True  → Judge 认为答案正确 (A. True) → 不弃权
    False → Judge 认为答案错误 (B. False) → 弃权
    None  → 解析失败
    """
    text = _strip_thinking_process(judge_raw).strip()
    # 精确匹配行首
    m = re.match(r"^\s*([AB])\b", text)
    if m:
        return m.group(1) == "A"
    # 匹配 "answer is A/B" / "答案是 True/False"
    m = re.search(r"(?:answer|答案)\s+(?:is\s+)?([AB]|True|False)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower()
        return val not in ("false", "b")
    # 匹配 "A. True" / "B. False"
    if re.search(r"\bA\.?\s*True\b", text, re.IGNORECASE):
        return True
    if re.search(r"\bB\.?\s*False\b", text, re.IGNORECASE):
        return False
    # 匹配纯文本 "True." / "False." (judge 直接输出句子)
    m = re.match(r"^(True|False)[.\s]", text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


# ═══════════════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════════════

def _get_proposed_answer(question: str, options: list, model_name: str) -> str:
    """获取 proposed answer 并解析出选项标签（最多重试 2 次）。"""
    mc_prompt = _format_mc_question(question, options) if options else question
    for attempt in range(2):
        prompt = f"Question: {mc_prompt}\nAnswer:"
        raw = call_llm(prompt, model_name=model_name)
        parsed = _parse_choice(raw)
        if not parsed:
            clean = _strip_thinking_process(raw)
            parsed = _parse_choice(clean)
        if parsed:
            return parsed
    return parsed or ""


def _cooperate_self(
    question: str,
    options: list,
    target_model: str,
    judge_model: str | None = None,
) -> dict:
    """COOPERATE-Self: 同一模型扮演 3 个领域专家。"""
    if judge_model is None:
        judge_model = target_model

    opts = options or []
    mc_prompt = _format_mc_question(question, opts) if opts else question

    # Step 1
    proposed = _get_proposed_answer(question, opts, target_model)
    logger.info(f"[COOP-Self] proposed_answer={proposed}")

    # Step 2: 3 个领域专家
    feedback_list = []
    for domain in COOPERATE_SELF_DOMAINS:
        # 2a: 生成领域知识
        full_k_prompt = (
            f"Question: {mc_prompt}\n"
            f"Proposed Answer: {proposed}\n\n"
            f"{COOPERATE_SELF_KNOWLEDGE.format(domain=domain)}"
        )
        knowledge_raw = call_llm(full_k_prompt, model_name=target_model)
        knowledge = _strip_thinking_process(knowledge_raw)

        # 2b: 领域专家审查
        review_prompt = COOPERATE_SELF_REVIEW.format(
            domain_knowledge=knowledge,
            question=mc_prompt,
            generated_answer=proposed,
        )
        feedback_raw = call_llm(review_prompt, model_name=target_model)
        feedback = _strip_thinking_process(feedback_raw)
        feedback_list.append(feedback)
        logger.info(f"[COOP-Self] domain={domain} | feedback={feedback[:80]}")

    # Step 3: Judge 汇总
    judge_prompt = COOPERATE_SELF_JUDGE.format(
        question=mc_prompt,
        generated_answer=proposed,
        feedback_1=feedback_list[0],
        feedback_2=feedback_list[1],
        feedback_3=feedback_list[2],
    )
    judge_raw = call_llm(judge_prompt, model_name=judge_model)
    judge_decision = judge_vote(judge_raw)
    abstain = (judge_decision is False)

    logger.info(
        f"[COOP-Self] judge_raw={judge_raw[:80]} | judge={judge_decision} → abstain={abstain}"
    )
    return {
        "abstain": abstain,
        "original_answer": proposed,
        "judge_decision": judge_decision,
        "judge_raw": judge_raw,
        "feedbacks": feedback_list,
        "domains": COOPERATE_SELF_DOMAINS,
    }


def _cooperate_others(
    question: str,
    options: list,
    target_model: str,
    reviewer_models: list,
    judge_model: str | None = None,
    fallback_model: str | None = None,
) -> dict:
    """COOPERATE-Others: 不同模型交叉审查。"""
    if judge_model is None:
        judge_model = reviewer_models[0] if reviewer_models else target_model

    opts = options or []
    mc_prompt = _format_mc_question(question, opts) if opts else question

    # Step 1
    proposed = _get_proposed_answer(question, opts, target_model)
    logger.info(f"[COOP-Others] proposed_answer={proposed}")

    # Step 2: 每个 reviewer（失败时自动切换到 fallback_model）
    feedback_list = []
    for i, reviewer in enumerate(reviewer_models):
        try:
            review_prompt = COOPERATE_OTHERS_REVIEW.format(
                question=mc_prompt,
                generated_answer=proposed,
            )
            feedback_raw = call_llm(review_prompt, model_name=reviewer)
            feedback = _strip_thinking_process(feedback_raw)
        except Exception as e:
            logger.warning(f"[COOP-Others] reviewer[{i}]={reviewer} failed ({e}), fallback to {fallback_model}")
            if fallback_model and fallback_model != reviewer:
                try:
                    feedback_raw = call_llm(review_prompt, model_name=fallback_model)
                    feedback = _strip_thinking_process(feedback_raw)
                except Exception:
                    feedback = f"(reviewer {reviewer} unavailable)"
            else:
                feedback = f"(reviewer {reviewer} unavailable)"
        feedback_list.append(feedback)
        logger.info(f"[COOP-Others] reviewer[{i}]={reviewer} | feedback={feedback[:80]}")

    # 不足 3 个 reviewer 时填充空字符串
    while len(feedback_list) < 3:
        feedback_list.append("(no feedback)")

    # Step 3: Judge 汇总
    judge_prompt = COOPERATE_OTHERS_JUDGE.format(
        question=mc_prompt,
        generated_answer=proposed,
        feedback_1=feedback_list[0],
        feedback_2=feedback_list[1],
        feedback_3=feedback_list[2],
    )
    judge_raw = call_llm(judge_prompt, model_name=judge_model)
    judge_decision = judge_vote(judge_raw)
    abstain = (judge_decision is False)

    logger.info(
        f"[COOP-Others] judge_raw={judge_raw[:80]} | judge={judge_decision} → abstain={abstain}"
    )
    return {
        "abstain": abstain,
        "original_answer": proposed,
        "judge_decision": judge_decision,
        "judge_raw": judge_raw,
        "feedbacks": feedback_list,
        "reviewer_models": reviewer_models,
    }


# ═══════════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════════

def run_cooperate(
    question: str,
    options: list,
    model_name: str | None = None,
    mode: str = "self",
    judge_model: str | None = None,
    model2: str | None = None,
) -> dict:
    if model_name is None:
        model_name = DEFAULT_MODEL

    if mode == "self":
        return _cooperate_self(question, options, target_model=model_name, judge_model=judge_model)
    elif mode == "others":
        if model2:
            reviewers = [model2]
        else:
            others = get_other_models(model_name)
            if not others:
                reviewers = [model_name]
            else:
                reviewers = others[:3]
        return _cooperate_others(
            question, options, target_model=model_name,
            reviewer_models=reviewers,
            judge_model=judge_model or model_name,
            fallback_model=model_name,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════
# 批量运行 & 测试入口
# ═══════════════════════════════════════════════════════════════════════

def run_cooperate_batch(
    dataset: list,
    model_name: str | None = None,
    mode: str = "self",
    judge_model: str | None = None,
    model2: str | None = None,
) -> list:
    """在数据集上运行 COOPERATE。"""
    results = []
    for i, sample in enumerate(dataset):
        q = sample["question"]
        opts = sample["options"]
        gold = sample["gold_answer"]
        logger.info(f"--- Sample {i+1}/{len(dataset)} ---")
        try:
            result = run_cooperate(q, opts, model_name=model_name, mode=mode, judge_model=judge_model, model2=model2)
            result["gold_answer"] = gold
            result["question_preview"] = q[:80]
            results.append(result)
        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            results.append({
                "abstain": False, "original_answer": "",
                "judge_decision": None, "judge_raw": "",
                "feedbacks": [], "error": str(e),
                "gold_answer": gold,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="COOPERATE Strategy Runner")
    parser.add_argument("--test", action="store_true", help="Run single-sample test")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples")
    parser.add_argument("--model", type=str, default=None, help="Target model")
    parser.add_argument("--model2", type=str, default=None,
                        help="Reviewer model for Others mode (default: auto-pick other registered model)")
    parser.add_argument("--mode", type=str, default="self",
                        choices=["self", "others"], help="Self or Others mode")
    parser.add_argument("--judge", type=str, default=None, help="Judge model")
    parser.add_argument("--both", action="store_true",
                        help="Run both self and others and compare")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <level>{message}</level>")

    model_name = args.model or DEFAULT_MODEL

    from loader import load_mmlu
    from evaluator import compute_metrics, print_metrics

    if args.test:
        q = "What is the capital of France?"
        opts = ["A: Paris", "B: London", "C: Berlin", "D: Madrid"]
        result = run_cooperate(q, opts, model_name=model_name, mode=args.mode, judge_model=args.judge)
        print(f"\n{'='*60}")
        print(f"  COOPERATE-{args.mode.upper()} Test Result")
        print(f"{'='*60}")
        print(f"  Question: {q}")
        print(f"  Original Answer: {result['original_answer']}")
        print(f"  Judge Decision: {result.get('judge_decision', '?')}")
        print(f"  Abstain: {result['abstain']}")
        print(f"\n  Feedbacks:")
        for i, fb in enumerate(result.get("feedbacks", [])):
            print(f"    [{i+1}] {fb[:120]}")
        print(f"\n  Judge Raw: {result.get('judge_raw', '')[:200]}")
        return

    if args.both:
        dataset = load_mmlu(limit=args.limit)
        results_self = run_cooperate_batch(dataset, model_name=model_name, mode="self", judge_model=args.judge)
        results_others = run_cooperate_batch(
            dataset, model_name=model_name, mode="others",
            judge_model=args.judge, model2=args.model2,
        )

        print(f"\n{'='*70}")
        print(f"  COOPERATE-Self vs COOPERATE-Others | Model: {model_name}")
        print(f"{'='*70}")

        # 逐条对比
        print(f"\n{'Idx':>3} | {'Gold':>5} | {'Self_abstain':>11} | {'Others_abstain':>13} | "
              f"{'Self_judge':>10} | {'Others_judge':>12} | {'Match?':>6}")
        print("─" * 78)
        for i, (s, o, d) in enumerate(zip(results_self, results_others, dataset)):
            match = "YES" if s["abstain"] == o["abstain"] else "DIFF"
            print(f"  {i+1:>3} | {d['gold_answer']:>5} | "
                  f"{str(s['abstain']):>11} | {str(o['abstain']):>13} | "
                  f"{str(s.get('judge_decision','?')):>10} | {str(o.get('judge_decision','?')):>12} | "
                  f"{match:>6}")

        # 指标对比
        for label, res in [("COOPERATE-Self", results_self), ("COOPERATE-Others", results_others)]:
            ab_dec = [r["abstain"] for r in res]
            a_cor = [r.get("original_answer", "") == s["gold_answer"]
            for r, s in zip(res, dataset)]
            m = compute_metrics(ab_dec, a_cor)
            print_metrics(m, label)
        return

    # 单模式运行
    dataset = load_mmlu(limit=args.limit)
    results = run_cooperate_batch(
        dataset, model_name=model_name, mode=args.mode,
        judge_model=args.judge, model2=args.model2,
    )
    ab_dec = [r["abstain"] for r in results]
    a_cor = [r.get("original_answer", "") == s["gold_answer"]
    for r, s in zip(results, dataset)]
    m = compute_metrics(ab_dec, a_cor)

    print(f"\n{'='*70}")
    print(f"  Dataset: {len(dataset)} samples | Model: {model_name}")
    print(f"  Mode: COOPERATE-{args.mode.upper()}")
    print(f"{'='*70}\n")

    for i, (r, s) in enumerate(zip(results, dataset)):
        parsed = r.get("original_answer", "")
        correct = parsed == s["gold_answer"]
        abstain = r["abstain"]
        status = "ABSTAIN" if abstain else f"ANSWER={parsed}"
        check = "OK" if correct else "XX"
        judge_d = r.get("judge_decision", "?")
        print(f"  [{i+1:2d}] {status:12s} | gold={s['gold_answer']} | {check} | judge={judge_d}")

    print()
    print_metrics(m, f"COOPERATE-{args.mode.upper()}")


if __name__ == "__main__":
    main()
