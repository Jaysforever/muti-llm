"""
main.py — AbstainQA 统一实验编排入口。

功能:
  - 统一 CLI: --strategy / --dataset / --limit / --model
  - 自动分发到 baselines / compete / cooperate 模块
  - 结果自动保存为 JSON
  - 实验后自动计算 4 项指标
  - 错误记录到 error_log.txt，单条失败不中断
  - 自动生成可视化图表

用法:
  python main.py --strategy cooperate_others --dataset mmlu --limit 5 --model deepseek
  python main.py --strategy all_baselines --dataset mmlu --limit 10 --model minimax
  python main.py --strategy compete --dataset mmlu --limit 20 --model deepseek
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 修复 Windows GBK 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

from loader import load_mmlu, load_hellaswag, load_knowledge_crosswords, load_propaganda
from config import call_llm, DEFAULT_MODEL, list_models
from baselines import BASELINE_REGISTRY, _parse_choice, _format_mc_question
from evaluator import compute_metrics, print_metrics

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_dataset(name: str, limit: int):
    loaders = {
        "mmlu": load_mmlu,
        "hellaswag": load_hellaswag,
        "k-crosswords": load_knowledge_crosswords,
        "propaganda": load_propaganda,
    }
    if name not in loaders:
        if name == "abstain_absolute":
            from loader_absolute import load_electionqa
            return load_electionqa(limit=limit)
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())} + abstain_absolute")
    return loaders[name](limit=limit)


def _get_direct_answer(question: str, options: list, model_name: str) -> str:
    mc = _format_mc_question(question, options) if options else question
    raw = call_llm(f"Question: {mc}\nAnswer:", model_name=model_name)
    parsed = _parse_choice(raw)
    if not parsed:
        from baselines import _strip_thinking_process
        parsed = _parse_choice(_strip_thinking_process(raw))
    return parsed or ""


def _make_result_filename(dataset: str, strategy: str) -> tuple[Path, Path]:
    ts = _timestamp()
    json_path = RESULTS_DIR / f"{dataset}_{strategy}_{ts}.json"
    err_path = RESULTS_DIR / f"error_log_{dataset}_{strategy}_{ts}.txt"
    return json_path, err_path


# ═══════════════════════════════════════════════════════════════════════
# 实验分发器
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(
    dataset: list[dict],
    strategy: str,
    model_name: str,
    limit: int,
) -> list[dict]:
    """
    根据 strategy 分发到对应的模块，返回每条数据的原始结果 dict。
    """
    if strategy in BASELINE_REGISTRY:
        return _run_baseline_experiment(dataset, strategy, model_name)
    elif strategy == "compete":
        return _run_compete_experiment(dataset, model_name, orchestrator=None)  # orchestrator set per sample
    elif strategy == "cooperate_self":
        return _run_cooperate_experiment(dataset, model_name, mode="self")
    elif strategy == "cooperate_others":
        return _run_cooperate_experiment(dataset, model_name, mode="others")
    elif strategy == "all_baselines":
        # all_baselines 内部会递归调用自己，逐个跑每个 baseline
        all_results = []
        for s in BASELINE_REGISTRY:
            logger.info(f"Running baseline: {s}")
            r = _run_baseline_experiment(dataset, s, model_name)
            all_results.extend([{"_strategy": s, **x} for x in r])
        return all_results
    elif strategy == "pipeline_abstain_retrieve":
        return _run_pipeline_experiment(dataset, model_name, limit)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _run_baseline_experiment(
    dataset: list[dict], strategy: str, model_name: str,
) -> list[dict]:
    fn = BASELINE_REGISTRY[strategy]
    results = []
    for i, sample in enumerate(dataset):
        q, opts, gold = sample["question"], sample["options"], sample["gold_answer"]
        t0 = time.time()
        try:
            res = fn(q, opts, model_name=model_name)
            parsed = res.get("parsed_answer", "")
            if not parsed:
                parsed = _get_direct_answer(q, opts, model_name)
        except Exception as e:
            logger.error(f"[{strategy}] sample {i} failed: {e}")
            parsed = ""
            res = {"abstain": False, "error": str(e)}
        elapsed = time.time() - t0
        results.append({
            "sample_id": i,
            "question": q[:200],
            "gold_answer": gold,
            "model_answer": parsed,
            "is_abstain": res.get("abstain", False),
            "elapsed_s": round(elapsed, 2),
            "strategy": strategy,
            "error": res.get("error", ""),
        })
    return results


def _run_compete_experiment(dataset: list[dict], model_name: str,
                            orch: str | None = None) -> list[dict]:
    from compete import run_compete_single
    from config import get_other_models
    results = []
    if orch:
        orch = orch
    else:
        others = get_other_models(model_name)
        orch = others[0] if others else model_name
    for i, sample in enumerate(dataset):
        q, opts, gold = sample["question"], sample["options"], sample["gold_answer"]
        t0 = time.time()
        try:
            res = run_compete_single(q, opts, model_name=model_name, k=3, orch_model=orch)
        except Exception:
            # 编排模型失败时回退到自挑战模式
            try:
                logger.warning(f"[compete] orch {orch} failed, falling back to self-challenge")
                res = run_compete_single(q, opts, model_name=model_name, k=3, orch_model=model_name)
            except Exception as e2:
                logger.error(f"[compete] sample {i} failed: {e2}")
                parsed = ""
                res = {"abstain": False, "error": str(e2)}
        parsed = res.get("original_answer", "")
        elapsed = time.time() - t0
        results.append({
            "sample_id": i,
            "question": q[:200],
            "gold_answer": gold,
            "model_answer": parsed,
            "is_abstain": res.get("abstain", False),
            "elapsed_s": round(elapsed, 2),
            "strategy": "compete",
            "shaken_count": res.get("shaken_count", 0),
            "conflict_log": res.get("conflict_log", []),
            "error": res.get("error", ""),
        })
    return results


def _run_cooperate_experiment(
    dataset: list[dict], model_name: str, mode: str,
) -> list[dict]:
    from cooperate import run_cooperate
    results = []
    for i, sample in enumerate(dataset):
        q, opts, gold = sample["question"], sample["options"], sample["gold_answer"]
        t0 = time.time()
        try:
            res = run_cooperate(q, opts, model_name=model_name, mode=mode)
            parsed = res.get("original_answer", "")
        except Exception as e:
            logger.error(f"[cooperate_{mode}] sample {i} failed: {e}")
            parsed = ""
            res = {"abstain": False, "error": str(e)}
        elapsed = time.time() - t0
        results.append({
            "sample_id": i,
            "question": q[:200],
            "gold_answer": gold,
            "model_answer": parsed,
            "is_abstain": res.get("abstain", False),
            "judge_decision": res.get("judge_decision"),
            "elapsed_s": round(elapsed, 2),
            "strategy": f"cooperate_{mode}",
            "feedbacks": res.get("feedbacks", []),
            "error": res.get("error", ""),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
# 评估 & 可视化
# ═══════════════════════════════════════════════════════════════════════

def _run_pipeline_experiment(dataset: list[dict], model_name: str, limit: int) -> list[dict]:
    """运行 Abstain-Retrieve-Abstain 两步 pipeline。"""
    from retriever import build_election_retriever
    from pipeline import run_pipeline_batch
    retriever = build_election_retriever()
    return run_pipeline_batch(dataset, model_name, "compete", retriever)


def evaluate_results(results: list[dict]) -> dict:
    """从 JSON 结果计算 4 项指标（兼容 standard 和 pipeline 格式）。"""
    ab_key = "is_abstain" if "is_abstain" in (results[0] if results else {}) else "final_abstain"
    ans_key = "model_answer" if "model_answer" in (results[0] if results else {}) else "final_answer"
    ab_dec = [r.get(ab_key, False) for r in results]
    a_cor = [
        (r.get(ans_key, "") == r["gold_answer"])
        if r.get(ans_key) else False
        for r in results
    ]
    return compute_metrics(ab_dec, a_cor)


def save_results(
    results: list[dict],
    dataset: str,
    strategy: str,
) -> Path:
    json_path, _ = _make_result_filename(dataset, strategy)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {json_path}")
    return json_path


def log_errors(results: list[dict], dataset: str, strategy: str) -> Path:
    _, err_path = _make_result_filename(dataset, strategy)
    errors = [r for r in results if r.get("error")]
    if errors:
        with open(err_path, "w", encoding="utf-8") as f:
            for r in errors:
                f.write(f"[sample {r['sample_id']}] {r['error']}\n")
        logger.warning(f"{len(errors)} errors logged to {err_path}")
    return err_path


def generate_charts(results: list[dict], dataset: str, strategy: str) -> Path:
    """生成汇总图表，保存到 results/analysis_{timestamp}.png"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping chart")
        return Path()

    json_path, _ = _make_result_filename(dataset, strategy)
    chart_path = RESULTS_DIR / f"analysis_{json_path.stem}.png"

    # 准备数据
    ab_dec = np.array([r["is_abstain"] for r in results], dtype=float)
    a_cor = np.array([
        1.0 if r["model_answer"] == r["gold_answer"] else 0.0
        if r["model_answer"] else np.nan for r in results
    ], dtype=float)
    elapsed = np.array([r["elapsed_s"] for r in results], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{dataset} / {strategy}", fontsize=13)

    # 1. R-Acc vs A-Acc 散点
    ax = axes[0]
    answered = ~ab_dec.astype(bool)
    answered_a_cor = a_cor[answered]
    answered_ab = ab_dec[answered]
    ax.scatter(answered_a_cor, answered_ab, alpha=0.6, s=60, c="steelblue")
    ax.set_xlabel("Reliable Accuracy (R-Acc)")
    ax.set_ylabel("Abstain Rate")
    ax.set_title("R-Acc vs Abstain Rate")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2. 每条数据的 abstain 分布
    ax = axes[1]
    colors = ["crimson" if r["is_abstain"] else ("green" if r["model_answer"] == r["gold_answer"] else "orange") for r in results]
    ax.bar(range(len(results)), [1 if r["is_abstain"] else 0 for r in results], color=colors, alpha=0.7)
    ax.set_xlabel("Sample ID")
    ax.set_ylabel("Abstain (1) / Answer (0)")
    ax.set_title("Abstain Decision per Sample")
    ax.set_ylim(-0.1, 1.3)

    # 3. 耗时分布
    ax = axes[2]
    ax.hist(elapsed, bins=10, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Latency Distribution (mean={elapsed.mean():.1f}s)")
    ax.axvline(elapsed.mean(), color="crimson", linestyle="--", label="mean")
    ax.legend()

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"Chart saved to {chart_path}")
    return chart_path


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AbstainQA Unified Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Strategies: compete, cooperate_self, cooperate_others, all_baselines, "
            "or any baseline name (self_reflect, more_info, gen_match, nota, "
            "token_prob, ask_calibrate, sc_threshold)\n"
            "Datasets: mmlu, hellaswag, k-crosswords, propaganda\n"
            "Examples:\n"
            "  python main.py --strategy cooperate_others --dataset mmlu --limit 5\n"
            "  python main.py --strategy all_baselines --dataset mmlu --limit 10\n"
            "  python main.py --strategy compete --dataset mmlu --limit 20 --model minimax\n"
        ),
    )
    parser.add_argument("--strategy", type=str, default="cooperate_others",
                        help="Strategy to run")
    parser.add_argument("--dataset", type=str, default="mmlu",
                        choices=["mmlu", "hellaswag", "k-crosswords", "propaganda", "abstain_absolute"],
                        help="Dataset")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of samples to evaluate")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <level>{message}</level>")

    model_name = args.model or DEFAULT_MODEL
    logger.info(f"Model: {model_name} | Dataset: {args.dataset} | Strategy: {args.strategy} | Limit: {args.limit}")

    # 加载数据
    dataset = _get_dataset(args.dataset, args.limit)

    # 运行实验
    logger.info(f"Running experiment on {len(dataset)} samples...")
    results = run_experiment(dataset, args.strategy, model_name, args.limit)

    # 保存 JSON
    json_path = save_results(results, args.dataset, args.strategy)

    # 错误日志
    err_path = log_errors(results, args.dataset, args.strategy)

    # 计算指标
    # 按 strategy 分组计算（all_baselines 会产生多个 strategy）
    strategy_groups: dict[str, list[dict]] = {}
    for r in results:
        s = r.get("_strategy", r.get("strategy", args.strategy))
        strategy_groups.setdefault(s, []).append(r)

    print(f"\n{'=' * 70}")
    print(f"  Experiment Summary")
    print(f"  Dataset: {args.dataset} | Strategy: {args.strategy} | Model: {model_name}")
    print(f"  Total samples: {len(results)} | Saved: {json_path}")
    if err_path.exists():
        print(f"  Errors: {err_path}")
    print(f"{'=' * 70}")

    for strat, strat_results in strategy_groups.items():
        metrics = evaluate_results(strat_results)
        print_metrics(metrics, strat)

    # 可视化
    if not args.no_chart:
        try:
            generate_charts(results, args.dataset, args.strategy)
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"  All results: {json_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
