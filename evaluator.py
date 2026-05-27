"""
evaluator.py — 计算 AbstainQA 四项评估指标。

指标定义（论文 Section 3，Figure 2）:
  A = 正确回答,  B = 正确但弃权
  C = 错误回答,  D = 错误但弃权

  Reliable Accuracy (R-Acc)  = A / (A + C)
  Effective Reliability (ER) = (A - C) / (A + B + C + D)
  Abstain Accuracy (A-Acc)   = (A + D) / (A + B + C + D)
  Abstain F1 (A-F1)          = HM(precision, recall)
    precision = D / (B + D),  recall = D / (C + D)
"""


def compute_metrics(
    abstain_decisions: list[bool],
    answer_correct: list[bool],
) -> dict[str, float]:
    """
    根据每条数据的弃权/正确标记计算 4 项指标。

    Parameters
    ----------
    abstain_decisions : list[bool]
        每条数据是否弃权 (True=弃权)。
    answer_correct : list[bool]
        每条数据若直接回答是否正确；弃权时可传任意值（不影响计数）。

    Returns
    -------
    dict 包含 R-Acc, ER, A-Acc, A-F1 以及 A, B, C, D 计数。
    """
    A = B = C = D = 0

    for abstain, correct in zip(abstain_decisions, answer_correct):
        if not abstain and correct:
            A += 1
        elif abstain and correct:
            B += 1
        elif not abstain and not correct:
            C += 1
        elif abstain and not correct:
            D += 1

    total = A + B + C + D
    if total == 0:
        return {"R-Acc": 0, "ER": 0, "A-Acc": 0, "A-F1": 0, "A": 0, "B": 0, "C": 0, "D": 0}

    r_acc = A / (A + C) if (A + C) > 0 else 0.0
    er = (A - C) / total
    a_acc = (A + D) / total

    precision = D / (B + D) if (B + D) > 0 else 0.0
    recall = D / (C + D) if (C + D) > 0 else 0.0
    a_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "R-Acc": round(r_acc, 4),
        "ER": round(er, 4),
        "A-Acc": round(a_acc, 4),
        "A-F1": round(a_f1, 4),
        "A": A, "B": B, "C": C, "D": D,
    }


def print_metrics(metrics: dict[str, float], method_name: str = "") -> None:
    """格式化打印指标。"""
    header = f" [{method_name}] " if method_name else " "
    print(f"{'=' * 60}")
    print(f"|{header.center(58)}|")
    print(f"{'=' * 60}")
    print(f"| A (正确回答)          | {metrics['A']:>5d}                          |")
    print(f"| B (正确但弃权)        | {metrics['B']:>5d}                          |")
    print(f"| C (错误回答)          | {metrics['C']:>5d}                          |")
    print(f"| D (错误但弃权)        | {metrics['D']:>5d}                          |")
    print(f"{'=' * 60}")
    print(f"| Reliable Accuracy     | {metrics['R-Acc']:.4f}                        |")
    print(f"| Effective Reliability | {metrics['ER']:.4f}                        |")
    print(f"| Abstain Accuracy      | {metrics['A-Acc']:.4f}                        |")
    print(f"| Abstain F1            | {metrics['A-F1']:.4f}                        |")
    print(f"{'=' * 60}\n")
