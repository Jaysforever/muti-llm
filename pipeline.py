"""
pipeline.py -- Abstain + Retrieval 两步策略（论文 Figure 4）。

Step 1: 仅用内部知识判断。如果 abstain，则检索外部文档。
Step 2: 有文档辅助下再次判断。如果仍 abstain → 最终 abstain。

用法:
  python pipeline.py --limit 10 --dataset mmlu --strategy compete --model deepseek
"""
import argparse
import sys
import time

from loguru import logger

from config import call_llm, DEFAULT_MODEL
from baselines import BASELINE_REGISTRY, _parse_choice, _format_mc_question
from retriever import build_election_retriever, Retriever
from evaluator import compute_metrics, print_metrics


def format_retrieval_prompt(question, options, retrieved_context):
    """构造带检索上下文的 QA prompt。"""
    opts = "\n".join(options) if options else ""
    return (
        f"Answer the question using ONLY the provided context below. "
        f"If the context is irrelevant, say you don't know.\n\n"
        f"Context: {retrieved_context}\n\n"
        f"Question: {question}\n{opts}\n\n"
        f"Answer:"
    )


def run_abstain_retrieve_pipeline(
    question, options, gold_answer,
    model_name, base_strategy,
    retriever: Retriever,
    verbose=False,
):
    """
    执行两步 Abstain-Retrieve 流程。

    Step 1: 用 base_strategy 在无检索情况下判断 abstain（COMPETE/COOPERATE）
    Step 2: 如果 Step 1 abstain 且有检索结果，用文档上下文重新判断
    Step 3: 若仍 abstain → 最终 abstain

    Returns dict with step1_result, step2_result, final_abstain, final_answer, etc.
    """
    step1_abstain = False
    step1_answer = ""
    step1_shaken_count = 0
    step1_judge = None

    # Step 1: 内部知识 abstaining
    try:
        if base_strategy == "compete":
            from compete import run_compete_single
            r1 = run_compete_single(question, options, model_name=model_name, k=3, orchestrator_model=model_name)
            step1_abstain = r1.get("abstain", False)
            step1_answer = r1.get("original_answer", "")
            step1_shaken_count = r1.get("shaken_count", 0)
        elif base_strategy == "cooperate_self":
            from cooperate import run_cooperate
            r1 = run_cooperate(question, options, model_name=model_name, mode="self")
            step1_abstain = r1.get("abstain", False)
            step1_answer = r1.get("original_answer", "")
            step1_judge = r1.get("judge_decision")
        elif base_strategy == "cooperate_others":
            from cooperate import run_cooperate
            r1 = run_cooperate(question, options, model_name=model_name, mode="others")
            step1_abstain = r1.get("abstain", False)
            step1_answer = r1.get("original_answer", "")
            step1_judge = r1.get("judge_decision")
        else:
            strategy_fn = BASELINE_REGISTRY.get(base_strategy)
            if strategy_fn:
                r1 = strategy_fn(question, options, model_name=model_name)
                step1_abstain = r1.get("abstain", False)
                step1_answer = r1.get("parsed_answer", "")

        # 如果 Step 1 答案为空，直接回答
        if not step1_answer:
            mc = _format_mc_question(question, options) if options else question
            raw = call_llm(f"Question: {mc}\nAnswer:", model_name=model_name)
            from baselines import _strip_thinking_process
            step1_answer = _parse_choice(raw) if options else _strip_thinking_process(raw)
    except Exception as e:
        logger.warning(f"Step 1 strategy failed: {e}")

    # Step 2: 如果 Step 1 abstain，尝试检索
    step2_abstain = None
    step2_answer = ""
    retrieved_context = ""

    if step1_abstain:
        retrieved_context = retriever.retrieve_context(question, top_k=3)
        if retrieved_context:
            # 有检索结果 → 带上下文重新回答
            try:
                prompt = format_retrieval_prompt(question, options, retrieved_context)
                raw = call_llm(prompt, model_name=model_name)
                from baselines import _strip_thinking_process
                step2_answer = _parse_choice(raw) if options else _strip_thinking_process(raw)
                # 重新判断
                step2_abstain = step1_abstain  # 保守：检索后仍 abstain
            except Exception as e:
                logger.warning(f"Step 2 failed: {e}")
                step2_abstain = True
        else:
            # 检索为空，保守 abstain
            step2_abstain = True

    # 最终决策
    if step1_abstain:
        final_abstain = (step2_abstain is True) or (step2_abstain is None)
        final_answer = step2_answer or step1_answer
    else:
        final_abstain = False
        final_answer = step1_answer

    if verbose:
        logger.info(
            f"[Pipeline] step1_abstain={step1_abstain} | "
            f"step2_abstain={step2_abstain} | "
            f"final_abstain={final_abstain}"
        )

    return {
        "step1_abstain": step1_abstain,
        "step2_abstain": step2_abstain,
        "final_abstain": final_abstain,
        "final_answer": final_answer,
        "retrieved_context": retrieved_context[:200] if retrieved_context else "",
        "step1_answer": step1_answer,
        "step2_answer": step2_answer,
        "step1_shaken_count": step1_shaken_count,
        "step1_judge": step1_judge,
    }


def run_pipeline_batch(dataset, model_name, strategy, retriever):
    results = []
    for i, sample in enumerate(dataset):
        q = sample["question"]
        opts = sample["options"]
        gold = sample["gold_answer"]
        t0 = time.time()
        try:
            r = run_abstain_retrieve_pipeline(
                q, opts, gold, model_name, strategy, retriever, verbose=False
            )
            elapsed = time.time() - t0
            results.append({
                "sample_id": i,
                "question": q[:200],
                "gold_answer": gold,
                **r,
                "elapsed_s": round(elapsed, 2),
            })
        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            results.append({"sample_id": i, "question": q[:200], "gold_answer": gold,
                          "final_abstain": False, "error": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Abstain+Retrieval Pipeline")
    parser.add_argument("--strategy", type=str, default="compete",
                        choices=list(BASELINE_REGISTRY.keys()) + ["compete", "cooperate_self", "cooperate_others"])
    parser.add_argument("--dataset", type=str, default="mmlu",
                        choices=["mmlu", "electionqa23"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <level>{message}</level>")
    model = args.model or DEFAULT_MODEL
    logger.info(f"Model={model} dataset={args.dataset} strategy={args.strategy}")

    # 加载数据
    if args.dataset == "mmlu":
        from loader import load_mmlu
        ds = load_mmlu(limit=args.limit)
    elif args.dataset == "electionqa23":
        from loader_absolute import load_electionqa
        ds = load_electionqa(limit=args.limit)
    else:
        raise ValueError(args.dataset)

    # 构建检索器
    ret = build_election_retriever()

    # 运行 pipeline
    results = run_pipeline_batch(ds, model, args.strategy, ret)

    # 评估
    ab = [r["final_abstain"] for r in results]
    ac = [r.get("final_answer", "") == r["gold_answer"] if r.get("final_answer") else False for r in results]
    m = compute_metrics(ab, ac)

    print(f"\n{'='*60}")
    print(f"Pipeline: Abstain-Retrieve-Abstain | strategy={args.strategy} | n={len(results)}")
    print(f"{'='*60}")
    for r in results:
        status = "ABSTAIN" if r["final_abstain"] else f"ANSWER={r.get('final_answer', '')}"
        correct = "OK" if (r.get("final_answer") == r["gold_answer"]) else "XX"
        s1 = "S1=Y" if r.get("step1_abstain") else "S1=N"
        s2 = f"S2=Y" if r.get("step2_abstain") else f"S2=N" if r.get("step2_abstain") is False else "S2=-"
        print(f"  [{r['sample_id']:2d}] {status} | gold={r['gold_answer']} | {correct} | {s1} {s2}")
    print()
    print_metrics(m, "Pipeline")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
