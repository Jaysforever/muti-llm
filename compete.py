"""compete.py -- COMPETE strategy (Table 13, k=3, shake_threshold support)."""
import random, re, sys, argparse, time
from loguru import logger
from config import call_llm, get_other_models, DEFAULT_MODEL
from baselines import _strip_thinking_process, _parse_choice, _format_mc_question
from prompts import (
    COMPETE_PROPOSED, COMPETE_ALTERNATIVE_ANSWER, COMPETE_KNOWLEDGE, COMPETE_CHALLENGE,
    COMPETE_PROPOSED_NO_REASONING, COMPETE_CHALLENGE_NO_REASONING,
)

def _get_alternative_answer_mc(original_answer, options, exclude_labels=None):
    label_to_text = {}
    for opt in options:
        m = re.match(r"^([A-D]):\s*(.+)", opt)
        if m: label_to_text[m.group(1)] = m.group(2)
    exclude = set(exclude_labels or [])
    available = [l for l in ["A","B","C","D"] if l != original_answer and l not in exclude]
    if not available: available = [l for l in ["A","B","C","D"] if l != original_answer]
    chosen = random.choice(available) if available else "A"
    return chosen, f"{chosen}: {label_to_text.get(chosen, chosen)}"

def _get_alternative_answer_open(question, options, proposed_answer, model_name):
    mc_prompt = _format_mc_question(question, options)
    raw = call_llm(COMPETE_ALTERNATIVE_ANSWER.format(question=mc_prompt, generated_answer=proposed_answer), model_name=model_name)
    return _strip_thinking_process(raw)[:200] or raw[:200]

def _generate_knowledge_paragraph(question, alternative_answer, model_name):
    return call_llm(COMPETE_KNOWLEDGE.format(question=question, alternative_answer=alternative_answer), model_name=model_name)

def _challenge(question, options, knowledge_paragraph, model_name, use_no_reasoning=False):
    mc_prompt = _format_mc_question(question, options) if options else question
    template = COMPETE_CHALLENGE_NO_REASONING if use_no_reasoning else COMPETE_CHALLENGE
    raw = call_llm(template.format(alternative_passage=knowledge_paragraph, question=mc_prompt), model_name=model_name)
    parsed = _parse_choice(raw) if options else _strip_thinking_process(raw)
    return raw, parsed

def _is_shaken(original, new, has_options):
    if not original or not new: return False
    if has_options: return original.strip().upper() != new.strip().upper()
    o, n = original.lower(), new.lower()
    if len(o) < 3 or len(n) < 3: return False
    return o not in n

def run_compete_single(question, options, model_name=None, k=3, orchestrator_model=None, shake_threshold=None):
    if model_name is None: model_name = DEFAULT_MODEL
    if orchestrator_model is None:
        others = get_other_models(model_name)
        orchestrator_model = others[0] if others else model_name
    opts = options or []
    has_options = len(opts) > 0
    mc_prompt = _format_mc_question(question, opts) if has_options else question

    use_no_reasoning = (shake_threshold is not None and shake_threshold <= 1)
    proposed_tmpl = COMPETE_PROPOSED_NO_REASONING if use_no_reasoning else COMPETE_PROPOSED

    # Step 1: proposed answer
    original_answer = ""
    for attempt in range(2):
        raw = call_llm(proposed_tmpl.format(question=mc_prompt), model_name=model_name)
        if has_options:
            original_answer = _parse_choice(raw)
            if not original_answer:
                original_answer = _parse_choice(_strip_thinking_process(raw))
            if not original_answer:
                m = re.search(r"(?:Answer|answer)\s*(?:is|:)?\s*([A-D])", _strip_thinking_process(raw), re.IGNORECASE)
                if m: original_answer = m.group(1)
        else:
            original_answer = _strip_thinking_process(raw)
        if original_answer: break
        logger.warning(f"[COMPETE] empty answer attempt {attempt+1}")

    logger.info(f"[COMPETE] Q: {question[:60]}... | ans={original_answer[:40] if original_answer else '(empty)'}")

    # Steps 2-5: k challenge rounds
    conflict_log, shaken_count, used_labels = [], 0, set()
    for round_idx in range(1, k+1):
        alt_label = ""
        if has_options:
            alt_label, alt_answer = _get_alternative_answer_mc(original_answer, opts, used_labels)
            used_labels.add(alt_label); alt_source = "random_choice"
        else:
            alt_answer = _get_alternative_answer_open(question, opts, str(original_answer), orchestrator_model)
            alt_source = "llm_generated"

        alt_knowledge = _generate_knowledge_paragraph(question, alt_answer, orchestrator_model)
        challenge_raw, new_answer = _challenge(question, opts, alt_knowledge, model_name, use_no_reasoning=use_no_reasoning)
        shaken = _is_shaken(original_answer, new_answer, has_options)
        if shaken: shaken_count += 1

        conflict_log.append({"round":round_idx,"alternative_label":alt_label if has_options else "","alternative_answer":alt_answer[:200],"alternative_source":alt_source,"knowledge_paragraph":alt_knowledge[:300],"challenge_response":challenge_raw[:200],"new_answer":new_answer[:200] if new_answer else "","shaken":shaken})
        logger.info(f"          r{round_idx}/{k}: alt={alt_label if has_options else alt_answer[:20]} | new={str(new_answer)[:20]} | shaken={shaken} | {shaken_count}/{round_idx}")

    abstain = shaken_count >= shake_threshold if shake_threshold is not None else shaken_count > (k/2)
    logger.info(f"          FINAL: shaken={shaken_count}/{k} (thresh={shake_threshold or int(k/2)+1}) -> abstain={abstain}")
    return {"abstain":abstain,"original_answer":original_answer,"shaken_count":shaken_count,"conflict_log":conflict_log}

# Batch & CLI omitted for brevity -- use main.py instead
