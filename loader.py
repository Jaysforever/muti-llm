"""
loader.py — 从本地 data/ 目录加载 AbstainQA 数据集。

数据来源: https://github.com/BunsenFeng/AbstainQA (作者官方)

所有数据集统一格式:
  {
    "dataset": "...",
    "dev":  [{question, choices: {A, B, C, D}, answer, id, ...}, ...],
    "test": [{question, choices: {A, B, C, D}, answer, id, ...}, ...],
  }
"""
import json
from pathlib import Path

from loguru import logger

DATA_DIR = Path(__file__).parent / "data"


def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_list_format(choices: dict[str, str]) -> list[str]:
    """将 {'A': 'xxx', 'B': 'yyy', ...} 转为 ['A: xxx', 'B: yyy', ...]"""
    return [f"{label}: {text}" for label, text in choices.items()]


def _parse_samples(raw: list[dict]) -> list[dict]:
    """将原始数据转换为项目统一格式。"""
    samples = []
    for item in raw:
        choices = item["choices"]
        samples.append({
            "question": item["question"],
            "options": _to_list_format(choices),
            "options_raw": list(choices.values()),
            "gold_answer": item["answer"],
            "gold_index": ord(item["answer"]) - ord("A"),
            "id": item.get("id", ""),
            "domain": item.get("domain", item.get("subject", "")),
        })
    return samples


def load_mmlu(limit: int | None = None, split: str = "test") -> list[dict]:
    """加载 MMLU 数据集。split='dev'|'test'"""
    logger.info(f"Loading MMLU ({split}), limit={limit}")
    data = _load_json("mmlu.json")
    raw = data[split]
    samples = _parse_samples(raw)
    if limit:
        samples = samples[:limit]
    logger.info(f"Loaded {len(samples)} MMLU samples")
    return samples


def load_hellaswag(limit: int | None = None, split: str = "test") -> list[dict]:
    """加载 Hellaswag 数据集。"""
    logger.info(f"Loading Hellaswag ({split}), limit={limit}")
    data = _load_json("hellaswag.json")
    raw = data[split]
    samples = _parse_samples(raw)
    if limit:
        samples = samples[:limit]
    logger.info(f"Loaded {len(samples)} Hellaswag samples")
    return samples


def load_knowledge_crosswords(limit: int | None = None, split: str = "test") -> list[dict]:
    """加载 Knowledge Crosswords 数据集。"""
    logger.info(f"Loading Knowledge Crosswords ({split}), limit={limit}")
    data = _load_json("knowledge_crosswords.json")
    raw = data[split]
    samples = _parse_samples(raw)
    if limit:
        samples = samples[:limit]
    logger.info(f"Loaded {len(samples)} Knowledge Crosswords samples")
    return samples


def load_propaganda(limit: int | None = None, split: str = "test") -> list[dict]:
    """加载 Propaganda 数据集。"""
    logger.info(f"Loading Propaganda ({split}), limit={limit}")
    data = _load_json("propaganda.json")
    raw = data[split]
    samples = _parse_samples(raw)
    if limit:
        samples = samples[:limit]
    logger.info(f"Loaded {len(samples)} Propaganda samples")
    return samples
