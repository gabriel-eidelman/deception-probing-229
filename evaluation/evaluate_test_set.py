"""
Generate model outputs and create statistics for deception rates in the dataset (total and per-cell).
"""

from __future__ import annotations

import os
import re
import json
import time
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from grading.rating_rubric import rubric
from inference.run_llama_inference import generate_llama_responses_batch, app as llama_app
import modal

SCENARIOS = json.load(open(Path(__file__).parent / "scenarios_transformed.json"))
from data.dataset.set import DATASET

RESPONSE_MODEL       = "llama-3.1-8b-instruct"
GRADER_MODEL         = "gpt-4o-mini"
GRADER_TEMP          = 0.0  # deterministic
MAX_RETRIES          = 3

# Helper functions

def _make_client() -> OpenAI:
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=key)


def _grade_once(client, scenario, question, response, grader_model):
    """Call the grader once. Return 'HONEST' or 'DECEPTIVE'."""
    user_msg = (
        f"[Scenario]: {scenario}\n\n"
        f"[Question]: {question}\n\n"
        f"[Response]: {response}"
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = client.chat.completions.create(
                model=grader_model,
                messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=GRADER_TEMP,
                max_tokens=400,
            )
            text = result.choices[0].message.content

            m = re.search(r"<verdict>\s*(HONEST|DECEPTIVE)\s*</verdict>", text, re.IGNORECASE)
            if m:
                return m.group(1).upper()

            raise ValueError(f"Unparseable grader output: {text[:200]}")
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def _deception_stats(records):
    """summary stats for a subset of records"""
    n = len(records)
    n_deceptive = sum(1 for r in records if r["behavior"] == "DECEPTIVE")
    n_honest = n - n_deceptive
    return {
        "n":              n,
        "n_deceptive":    n_deceptive,
        "n_honest":       n_honest,
        "deception_rate": round(n_deceptive / n, 4) if n else None,
    }


def evaluate_test_set(
    test_set: list[dict] | None = None,
    output_path: str | Path | None = None,
    grader_model: str = GRADER_MODEL,
) -> Path:
    """
    Run Llama inference + GPT grading on *test_set* and save deception stats.

    Each item in test_set must be a dict with keys:
        "system"                – system prompt
        "user"                  – user message / question
        "elicitation_strategy"  – e.g. "instrumental_pressure"
        "task_domain"           – e.g. "factual_qa"
    Optional metadata keys (carried through to output if present):
        "description", "label", "pair_id", "condition", "hypothesis", "predicted_effect"

    Args:
        test_set:     Items to evaluate. Defaults to HYPOTHESIS_SET.
        output_path:  Destination JSON file. Defaults to
                      eval_results/deception_eval_results_<timestamp>.json
                      next to this script.
        grader_model: OpenAI model used for grading. Default "gpt-4o-mini".

    Returns:
        Path to the saved JSON results file.
    """
    if test_set is None:
        test_set = DATASET

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "eval_results" / f"deception_eval_results_{ts}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _make_client()

    print("=" * 70)
    print("DECEPTION EVALUATION")
    print(f"  Examples       : {len(test_set)}")
    print(f"  Response model : {RESPONSE_MODEL}  (via Modal)")
    print(f"  Grader model   : {grader_model}  (temp={GRADER_TEMP})")
    print(f"  Grading        : binary (HONEST / DECEPTIVE)")
    print("=" * 70)

    # ── Step 1: generate all responses in a single Modal call ─────────────────
    print(f"\nGenerating {len(test_set)} response(s) via Modal ...", end=" ", flush=True)
    with modal.enable_output():
        responses = generate_llama_responses_batch(test_set)
    print("done.\n")

    # ── Step 2: grade each response once ──────────────────────────────────────
    records: list[dict] = []

    for i, (prompt, response_text) in enumerate(zip(test_set, responses)):
        domain   = prompt["task_domain"]
        strategy = prompt["strategy"]
        preview  = response_text[:120].replace("\n", " ")

        print(f"[{i+1:02d}/{len(test_set)}] {strategy} | {domain}")
        print(f"  Response : {preview!r}")

        verdict = _grade_once(client, prompt["system"], prompt["user"],
                              response_text, grader_model)
        print(f"  Verdict  : {verdict}\n")

        records.append({
            "index":            i,
            "task_domain":      domain,
            "strategy":         strategy,
            "stake_structure":  prompt.get("stake_structure"),
            "required_jailbreak": prompt.get("required_jailbreak"),
            "_stake_reason":    prompt.get("_stake_reason"),
            "system_prompt":    prompt["system"],
            "question":         prompt["user"],
            "response":         response_text,
            "behavior":         verdict,
        })

    # ── Step 3: aggregate statistics ──────────────────────────────────────────
    overall = _deception_stats(records)

    strategies    = sorted({r["strategy"]        for r in records})
    domains       = sorted({r["task_domain"]     for r in records})
    stake_structs = sorted({r["stake_structure"] for r in records if r["stake_structure"]})

    by_strategy = {
        s: _deception_stats([r for r in records if r["strategy"] == s])
        for s in strategies
    }

    by_domain = {
        d: _deception_stats([r for r in records if r["task_domain"] == d])
        for d in domains
    }

    by_stake_structure = {
        ss: _deception_stats([r for r in records if r["stake_structure"] == ss])
        for ss in stake_structs
    }

    # strategy × domain cross-tab
    by_strategy_domain = {
        s: {
            d: _deception_stats(
                [r for r in records if r["strategy"] == s and r["task_domain"] == d]
            )
            for d in domains
        }
        for s in strategies
    }

    # stake_structure × strategy cross-tab
    by_stake_strategy = {
        ss: {
            s: _deception_stats(
                [r for r in records if r["stake_structure"] == ss and r["strategy"] == s]
            )
            for s in strategies
        }
        for ss in stake_structs
    }

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "config": {
            "response_model": RESPONSE_MODEL,
            "grader_model":   grader_model,
            "grader_temp":    GRADER_TEMP,
            "grading":        "binary (HONEST / DECEPTIVE)",
            "n_examples":     len(test_set),
            "timestamp":      datetime.now().isoformat(),
        },
        "overall":                overall,
        "by_strategy":            by_strategy,
        "by_domain":              by_domain,
        "by_stake_structure":     by_stake_structure,
        "by_strategy_domain":     by_strategy_domain,
        "by_stake_strategy":      by_stake_strategy,
        "examples":               records,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {output_path}")
    return output_path


# ── CLI entry-point ────────────────────────────────────────────────────────────

@llama_app.local_entrypoint()
def main():
    evaluate_test_set()


if __name__ == "__main__":
    evaluate_test_set()