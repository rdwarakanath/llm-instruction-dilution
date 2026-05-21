"""
structured_eval.py
------------------
Evaluates model output for the structured text generation task
(emails and short reports).

Evaluation dimensions:
  1. format_compliance  : Are required sections present?
  2. rouge_l_f1         : Lexical overlap with reference output
  3. length_ok          : Is output within expected word count range?
  4. no_markdown_fence  : Model should not wrap output in code fences
  5. no_preamble        : Model should not add explanation before output

Final score = weighted combination (defined in WEIGHTS below).

Rules:
  - Imports only from metrics.py and stdlib
  - Never modifies log records
  - Returns a new dict with all scores
"""

import re
from typing import Optional

from metrics import format_compliance, rouge_l

# ─────────────────────────────────────────────────────────────
# SCORING WEIGHTS — must sum to 1.0
# ─────────────────────────────────────────────────────────────
WEIGHTS = {
    "format_compliance": 0.45,  # primary: correct structure
    "rouge_l_f1":        0.35,  # secondary: content overlap with reference
    "length_ok":         0.10,  # tertiary: appropriate length
    "clean_output":      0.10,  # no fences, no preamble
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Word count ranges by output type
LENGTH_RANGES = {
    "email":  (60, 200),
    "report": (80, 300),
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _detect_preamble(response_text: str) -> bool:
    """
    Returns True if the model added an explanation before the output.
    Common preamble patterns to detect.
    """
    text = response_text.strip().lower()
    preamble_patterns = [
        r"^here is",
        r"^here'?s",
        r"^sure",
        r"^certainly",
        r"^of course",
        r"^below is",
        r"^as requested",
        r"^i have written",
        r"^i've written",
        r"^please find",
        r"^the following",
    ]
    for pattern in preamble_patterns:
        if re.match(pattern, text):
            return True
    return False


def _word_count(text: str) -> int:
    """Returns the word count of a string."""
    return len(text.split()) if text else 0


def _check_length(response_text: str, output_type: str) -> dict:
    """
    Checks if the response is within the expected word count range.

    Args:
        response_text: Model output.
        output_type:   'email' or 'report'

    Returns:
        {'ok': bool, 'word_count': int, 'min': int, 'max': int}
    """
    wc = _word_count(response_text)
    min_w, max_w = LENGTH_RANGES.get(output_type, (50, 400))
    return {
        "ok":         min_w <= wc <= max_w,
        "word_count": wc,
        "min":        min_w,
        "max":        max_w,
    }


# ─────────────────────────────────────────────────────────────
# CORE EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_struct_response(
    response_text: str,
    reference_output: str,
    required_sections: list,
    output_type: str = "email",
    record_id: str = "",
) -> dict:
    """
    Evaluates a single model response for the structured text task.

    Args:
        response_text:     Raw model output string.
        reference_output:  Reference output from the dataset.
        required_sections: List of section names, e.g. ['subject','greeting','body','sign_off']
        output_type:       'email' or 'report'
        record_id:         Dataset record ID for traceability.

    Returns:
        {
            'record_id':          str,
            'format_score':       float,   # format_compliance score
            'sections_found':     list,
            'sections_missing':   list,
            'rouge_l_precision':  float,
            'rouge_l_recall':     float,
            'rouge_l_f1':         float,
            'word_count':         int,
            'length_ok':          bool,
            'has_markdown_fence': bool,
            'has_preamble':       bool,
            'clean_output_score': float,   # 1.0 if no fence or preamble
            'final_score':        float,
            'failure_reason':     str,
        }
    """
    if not response_text or not response_text.strip():
        return {
            "record_id":          record_id,
            "format_score":       0.0,
            "sections_found":     [],
            "sections_missing":   required_sections,
            "rouge_l_precision":  0.0,
            "rouge_l_recall":     0.0,
            "rouge_l_f1":         0.0,
            "word_count":         0,
            "length_ok":          False,
            "has_markdown_fence": False,
            "has_preamble":       False,
            "clean_output_score": 0.0,
            "final_score":        0.0,
            "failure_reason":     "Empty response from model.",
        }

    # ── 1. Format compliance ──
    fc = format_compliance(response_text, required_sections)
    format_score = fc["score"]

    # ── 2. ROUGE-L ──
    rl = rouge_l(response_text, reference_output)

    # ── 3. Length check ──
    length = _check_length(response_text, output_type)
    length_score = 1.0 if length["ok"] else 0.5  # partial credit if close

    # ── 4. Clean output ──
    has_fence    = fc["has_markdown_fence"]
    has_preamble = _detect_preamble(response_text)

    # Each violation costs 0.5 points; both = 0.0
    clean_score = 1.0
    if has_fence:
        clean_score -= 0.5
    if has_preamble:
        clean_score -= 0.5
    clean_score = max(0.0, clean_score)

    # ── 5. Weighted final score ──
    final_score = (
        WEIGHTS["format_compliance"] * format_score +
        WEIGHTS["rouge_l_f1"]        * rl["f1"] +
        WEIGHTS["length_ok"]         * length_score +
        WEIGHTS["clean_output"]      * clean_score
    )
    final_score = round(max(0.0, min(1.0, final_score)), 4)

    # ── 6. Failure reasons ──
    reasons = []
    if fc["sections_missing"]:
        reasons.append(f"Missing sections: {fc['sections_missing']}")
    if has_fence:
        reasons.append("Used markdown fences.")
    if has_preamble:
        reasons.append("Added preamble before output.")
    if not length["ok"]:
        reasons.append(
            f"Length {length['word_count']}w outside range "
            f"[{length['min']},{length['max']}]."
        )

    return {
        "record_id":          record_id,
        "format_score":       round(format_score, 4),
        "sections_found":     fc["sections_found"],
        "sections_missing":   fc["sections_missing"],
        "rouge_l_precision":  rl["precision"],
        "rouge_l_recall":     rl["recall"],
        "rouge_l_f1":         rl["f1"],
        "word_count":         length["word_count"],
        "length_ok":          length["ok"],
        "has_markdown_fence": has_fence,
        "has_preamble":       has_preamble,
        "clean_output_score": round(clean_score, 4),
        "final_score":        final_score,
        "failure_reason":     " | ".join(reasons),
    }


# ─────────────────────────────────────────────────────────────
# BATCH EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_struct_batch(log_records: list) -> list:
    """
    Evaluates a batch of log records for the structured text task.

    Args:
        log_records: List of log record dicts from logger.load_run_logs().

    Returns:
        List of evaluation result dicts.
    """
    results = []

    for record in log_records:
        if record.get("task_type") != "struct_task":
            continue

        record_id        = record.get("record_id", "")
        response_text    = record.get("response_text", "")
        ground_truth     = record.get("ground_truth", "")
        success          = record.get("success", False)

        # Determine output_type and required_sections from ground_truth metadata
        # Ground truth for struct_task is a string (reference output)
        # Sections come from the dataset — we store them in ground_truth as
        # a JSON-encoded dict when logging, or use defaults
        required_sections = ["subject", "greeting", "body", "sign_off"]
        output_type       = "email"

        # Try to parse structured ground truth if it was stored as JSON
        if isinstance(ground_truth, str):
            try:
                import json as _json
                gt_parsed = _json.loads(ground_truth)
                if isinstance(gt_parsed, dict):
                    required_sections = gt_parsed.get(
                        "sections_required", required_sections
                    )
                    output_type = gt_parsed.get("type", output_type)
                    reference   = gt_parsed.get("reference_output", ground_truth)
                else:
                    reference = ground_truth
            except Exception:
                reference = ground_truth
        else:
            reference = str(ground_truth) if ground_truth else ""

        if not success or not response_text:
            result = {
                "record_id":          record_id,
                "format_score":       0.0,
                "sections_found":     [],
                "sections_missing":   required_sections,
                "rouge_l_precision":  0.0,
                "rouge_l_recall":     0.0,
                "rouge_l_f1":         0.0,
                "word_count":         0,
                "length_ok":          False,
                "has_markdown_fence": False,
                "has_preamble":       False,
                "clean_output_score": 0.0,
                "final_score":        0.0,
                "failure_reason":     f"Model call failed: {record.get('error','unknown')}",
            }
        else:
            result = evaluate_struct_response(
                response_text    = response_text,
                reference_output = reference,
                required_sections= required_sections,
                output_type      = output_type,
                record_id        = record_id,
            )

        # Enrich with experiment metadata
        result["run_id"]       = record.get("run_id", "")
        result["model_id"]     = record.get("model_id", "")
        result["strategy"]     = record.get("strategy", "")
        result["noise_level"]  = record.get("noise_level", "")
        result["temperature"]  = record.get("temperature", 0.0)
        result["repeat_index"] = record.get("repeat_index", 1)
        result["latency_ms"]   = record.get("latency_ms", 0)
        result["output_type"]  = output_type

        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python src/structured_eval.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== structured_eval.py self-test ===\n")

    reference = """Subject: Request for Project Deadline Extension

Dear Arjun,

I hope this email finds you well. I am writing to request a one-week extension on the current project deadline.

Unfortunately, one of our key team members has been on medical leave this week, which has impacted our progress significantly.

I sincerely apologise for any inconvenience this may cause and assure you that we will utilise the additional time effectively.

Thank you for your understanding.

Best regards,
Priya Nair"""

    # 1. Perfect response (same as reference)
    r = evaluate_struct_response(
        response_text    = reference,
        reference_output = reference,
        required_sections= ["subject", "greeting", "body", "sign_off"],
        output_type      = "email",
        record_id        = "struct_001",
    )
    assert r["format_score"]  == 1.0
    assert r["rouge_l_f1"]    == 1.0
    assert r["length_ok"]     is True
    assert r["final_score"]   == 1.0
    print(f"[1] Perfect response — final_score={r['final_score']} ✓")

    # 2. Missing sign_off
    no_signoff = """Subject: Request for Extension

Dear Arjun,

I need more time for the project due to a team member being on leave.
"""
    r2 = evaluate_struct_response(
        response_text    = no_signoff,
        reference_output = reference,
        required_sections= ["subject", "greeting", "body", "sign_off"],
        output_type      = "email",
        record_id        = "struct_002",
    )
    assert "sign_off" in r2["sections_missing"]
    assert r2["format_score"] < 1.0
    print(f"[2] Missing sign_off — format_score={r2['format_score']}, missing={r2['sections_missing']} ✓")

    # 3. Markdown fence penalised
    fenced = f"```\n{reference}\n```"
    r3 = evaluate_struct_response(
        response_text    = fenced,
        reference_output = reference,
        required_sections= ["subject", "greeting", "body", "sign_off"],
        output_type      = "email",
        record_id        = "struct_003",
    )
    assert r3["has_markdown_fence"] is True
    assert r3["clean_output_score"] < 1.0
    print(f"[3] Markdown fence — clean_score={r3['clean_output_score']} ✓")

    # 4. Preamble detected
    preamble_resp = "Here is the email you requested:\n\n" + reference
    r4 = evaluate_struct_response(
        response_text    = preamble_resp,
        reference_output = reference,
        required_sections= ["subject", "greeting", "body", "sign_off"],
        output_type      = "email",
        record_id        = "struct_004",
    )
    assert r4["has_preamble"] is True
    assert r4["clean_output_score"] < 1.0
    print(f"[4] Preamble detected — clean_score={r4['clean_output_score']} ✓")

    # 5. Empty response
    r5 = evaluate_struct_response(
        response_text    = "",
        reference_output = reference,
        required_sections= ["subject", "greeting", "body", "sign_off"],
        output_type      = "email",
        record_id        = "struct_005",
    )
    assert r5["final_score"] == 0.0
    print(f"[5] Empty response — final_score={r5['final_score']} ✓")

    # 6. Report type
    report_resp = """Progress Report – Customer Portal

Overall Status: On Track

Progress This Period:
UI redesign completed. Backend at 60%.

Blockers:
API integration delayed by vendor.

Next Steps:
Complete backend by mid-April."""

    r6 = evaluate_struct_response(
        response_text    = report_resp,
        reference_output = report_resp,
        required_sections= ["title", "status", "progress", "blockers", "next_steps"],
        output_type      = "report",
        record_id        = "struct_006",
    )
    assert r6["rouge_l_f1"]  == 1.0
    print(f"[6] Report type — format_score={r6['format_score']}, rouge={r6['rouge_l_f1']} ✓")

    # 7. Partial overlap ROUGE
    partial = """Subject: Extension Request

Dear Arjun,

I need an extension due to team illness.

Regards,
Priya"""
    r7 = evaluate_struct_response(
        response_text    = partial,
        reference_output = reference,
        required_sections= ["subject", "greeting", "body", "sign_off"],
        output_type      = "email",
        record_id        = "struct_007",
    )
    assert 0.0 < r7["rouge_l_f1"] < 1.0
    print(f"[7] Partial overlap — rouge_l_f1={r7['rouge_l_f1']} ✓")

    # 8. Batch evaluator
    log_records = [
        {
            "task_type":     "struct_task",
            "record_id":     "struct_001",
            "response_text": reference,
            "ground_truth":  reference,
            "success":       True,
            "run_id":        "test_run",
            "model_id":      "model_3b",
            "strategy":      "modular",
            "noise_level":   "short",
            "temperature":   0.0,
            "repeat_index":  1,
            "latency_ms":    980,
            "error":         "",
        },
        {
            "task_type":     "struct_task",
            "record_id":     "struct_002",
            "response_text": "",
            "ground_truth":  reference,
            "success":       False,
            "run_id":        "test_run",
            "model_id":      "model_3b",
            "strategy":      "modular",
            "noise_level":   "short",
            "temperature":   0.0,
            "repeat_index":  1,
            "latency_ms":    0,
            "error":         "Timeout",
        },
    ]
    batch = evaluate_struct_batch(log_records)
    assert len(batch) == 2
    assert batch[0]["final_score"] == 1.0
    assert batch[1]["final_score"] == 0.0
    print(f"[8] Batch evaluator — scores={[b['final_score'] for b in batch]} ✓")

    print("\n✅  All structured_eval.py tests passed.")
