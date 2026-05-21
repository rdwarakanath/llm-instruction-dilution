"""
json_eval.py
------------
Evaluates model output for the JSON extraction task.

Takes a log record (from logger.py) and its ground truth,
runs all metrics, and returns a structured evaluation result.

Evaluation dimensions:
  1. json_valid       : Did the model return valid JSON?
  2. field_precision  : Of the fields returned, how many are correct?
  3. field_recall     : Of the expected fields, how many were found?
  4. field_f1         : Harmonic mean of precision and recall
  5. hallucination    : Did the model invent fields not in ground truth?
  6. exact_match      : Did the full output exactly match ground truth?
  7. format_penalty   : Did the model use markdown fences? (should not)

Final score = weighted combination (defined in WEIGHTS below).

Rules:
  - Imports only from metrics.py and stdlib
  - Never modifies the log record — only reads it
  - Returns a new dict with all scores
"""

import json
from typing import Optional

from metrics import (
    exact_match,
    field_f1,
    field_precision,
    field_recall,
    hallucination_rate,
    json_validity,
    normalise_text,
)

# ─────────────────────────────────────────────────────────────
# SCORING WEIGHTS
# These define what matters most for the JSON task.
# Must sum to 1.0.
# ─────────────────────────────────────────────────────────────
WEIGHTS = {
    "field_f1":       0.50,   # primary metric — correct fields
    "field_recall":   0.20,   # penalise missing fields
    "json_valid":     0.20,   # must return parseable JSON
    "no_hallucination": 0.10, # penalise invented fields
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ─────────────────────────────────────────────────────────────
# CORE EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_json_response(
    response_text: str,
    expected_output: dict,
    record_id: str = "",
) -> dict:
    """
    Evaluates a single model response for the JSON extraction task.

    Args:
        response_text:   Raw model output string.
        expected_output: Ground truth dict from dataset record.
        record_id:       Dataset record ID (for traceability).

    Returns:
        {
            'record_id':        str,
            'json_valid':       bool,
            'parsed_output':    dict | None,
            'field_precision':  float,
            'field_recall':     float,
            'field_f1':         float,
            'hallucination':    float,
            'exact_match':      float,
            'format_penalty':   float,   # 0.0 = clean, 0.1 = had fences
            'final_score':      float,   # weighted composite
            'fields_correct':   list,    # keys that matched
            'fields_missing':   list,    # expected keys not in output
            'fields_extra':     list,    # output keys not in expected
            'failure_reason':   str,     # empty if fully correct
        }
    """
    # ── Step 1: validate JSON ──
    validity = json_validity(response_text)
    is_valid  = validity["valid"]
    parsed    = validity["parsed"]  # None if not valid

    # ── Step 2: field-level metrics ──
    if is_valid and parsed:
        prec  = field_precision(parsed, expected_output)
        rec   = field_recall(parsed, expected_output)
        f1    = field_f1(parsed, expected_output)
        hall  = hallucination_rate(parsed, expected_output)

        # Which fields matched / missed / extra
        fields_correct = [
            k for k in expected_output
            if k in parsed and normalise_text(str(parsed[k])) == normalise_text(str(expected_output[k]))
        ]
        fields_missing = [k for k in expected_output if k not in parsed]
        fields_extra   = [k for k in parsed if k not in expected_output]

    else:
        prec = rec = f1 = 0.0
        hall = 0.0
        fields_correct = []
        fields_missing = list(expected_output.keys())
        fields_extra   = []

    # ── Step 3: exact match (full JSON) ──
    em = 0.0
    if is_valid and parsed:
        pred_str = json.dumps(parsed, sort_keys=True)
        exp_str  = json.dumps(expected_output, sort_keys=True)
        em = exact_match(pred_str, exp_str)

    # ── Step 4: format penalty ──
    # Model was told not to use markdown fences
    format_penalty = 0.1 if (response_text and "```" in response_text) else 0.0

    # ── Step 5: weighted final score ──
    json_valid_score    = 1.0 if is_valid else 0.0
    no_hallucination_score = 1.0 - hall

    # If JSON is invalid, the response is a complete failure for this task.
    # No partial credit — a model that cannot return parseable JSON has
    # fundamentally failed the instruction regardless of other factors.
    if not is_valid:
        final_score = 0.0
    else:
        final_score = (
            WEIGHTS["field_f1"]         * f1 +
            WEIGHTS["field_recall"]     * rec +
            WEIGHTS["json_valid"]       * json_valid_score +
            WEIGHTS["no_hallucination"] * no_hallucination_score
        ) - format_penalty

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))

    # ── Step 6: failure reason ──
    reasons = []
    if not is_valid:
        reasons.append(f"Invalid JSON: {validity['reason']}")
    if fields_missing:
        reasons.append(f"Missing fields: {fields_missing}")
    if fields_extra:
        reasons.append(f"Hallucinated fields: {fields_extra}")
    if format_penalty > 0:
        reasons.append("Used markdown fences.")

    return {
        "record_id":       record_id,
        "json_valid":      is_valid,
        "parsed_output":   parsed,
        "field_precision": round(prec, 4),
        "field_recall":    round(rec, 4),
        "field_f1":        round(f1, 4),
        "hallucination":   round(hall, 4),
        "exact_match":     round(em, 4),
        "format_penalty":  round(format_penalty, 4),
        "final_score":     round(final_score, 4),
        "fields_correct":  fields_correct,
        "fields_missing":  fields_missing,
        "fields_extra":    fields_extra,
        "failure_reason":  " | ".join(reasons),
    }


# ─────────────────────────────────────────────────────────────
# BATCH EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_json_batch(log_records: list) -> list:
    """
    Evaluates a batch of log records for the JSON task.

    Args:
        log_records: List of log record dicts from logger.load_run_logs().
                     Each must have 'response_text', 'ground_truth', 'record_id'.

    Returns:
        List of evaluation result dicts, one per log record.
        Failed runs (success=False) get zero scores with reason noted.
    """
    results = []

    for record in log_records:
        if record.get("task_type") != "json_task":
            continue

        record_id     = record.get("record_id", "")
        response_text = record.get("response_text", "")
        ground_truth  = record.get("ground_truth")
        success       = record.get("success", False)

        # Parse ground truth string back to dict if needed
        if isinstance(ground_truth, str):
            try:
                expected = json.loads(ground_truth)
            except (json.JSONDecodeError, TypeError):
                expected = {}
        elif isinstance(ground_truth, dict):
            expected = ground_truth
        else:
            expected = {}

        if not success or not response_text:
            # Model call failed — zero score
            result = {
                "record_id":       record_id,
                "json_valid":      False,
                "parsed_output":   None,
                "field_precision": 0.0,
                "field_recall":    0.0,
                "field_f1":        0.0,
                "hallucination":   0.0,
                "exact_match":     0.0,
                "format_penalty":  0.0,
                "final_score":     0.0,
                "fields_correct":  [],
                "fields_missing":  list(expected.keys()) if expected else [],
                "fields_extra":    [],
                "failure_reason":  f"Model call failed: {record.get('error', 'unknown')}",
            }
        else:
            result = evaluate_json_response(response_text, expected, record_id)

        # Enrich with experiment metadata for downstream aggregation
        result["run_id"]      = record.get("run_id", "")
        result["model_id"]    = record.get("model_id", "")
        result["strategy"]    = record.get("strategy", "")
        result["noise_level"] = record.get("noise_level", "")
        result["temperature"] = record.get("temperature", 0.0)
        result["repeat_index"]= record.get("repeat_index", 1)
        result["latency_ms"]  = record.get("latency_ms", 0)

        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python src/json_eval.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== json_eval.py self-test ===\n")

    expected = {
        "name": "Ravi Kumar",
        "age": 21,
        "profession": "engineering student",
        "city": "Chennai",
        "college": "Anna University",
        "email": "ravi.kumar21@gmail.com"
    }

    # 1. Perfect response
    perfect = '{"name": "Ravi Kumar", "age": 21, "profession": "engineering student", "city": "Chennai", "college": "Anna University", "email": "ravi.kumar21@gmail.com"}'
    r = evaluate_json_response(perfect, expected, "json_001")
    assert r["json_valid"]      is True
    assert r["field_f1"]        == 1.0
    assert r["hallucination"]   == 0.0
    assert r["final_score"]     == 1.0
    print(f"[1] Perfect response — final_score={r['final_score']} ✓")

    # 2. Partial response — missing fields
    partial = '{"name": "Ravi Kumar", "age": 21}'
    r2 = evaluate_json_response(partial, expected, "json_001")
    assert r2["json_valid"]     is True
    assert r2["field_recall"]   < 1.0
    assert r2["fields_missing"] == ["profession", "city", "college", "email"]
    print(f"[2] Partial response — recall={r2['field_recall']}, missing={r2['fields_missing']} ✓")

    # 3. Hallucinated field
    hall = '{"name": "Ravi Kumar", "age": 21, "hobby": "cricket"}'
    r3 = evaluate_json_response(hall, expected, "json_001")
    assert r3["hallucination"]  > 0.0
    assert "hobby" in r3["fields_extra"]
    print(f"[3] Hallucination — rate={r3['hallucination']}, extra={r3['fields_extra']} ✓")

    # 4. Invalid JSON
    invalid = "Sorry, I cannot extract that."
    r4 = evaluate_json_response(invalid, expected, "json_001")
    assert r4["json_valid"]     is False
    assert r4["final_score"]    == 0.0
    print(f"[4] Invalid JSON — final_score={r4['final_score']} ✓")

    # 5. Markdown fence — penalised
    fenced = '```json\n{"name": "Ravi Kumar", "age": 21}\n```'
    r5 = evaluate_json_response(fenced, expected, "json_001")
    assert r5["json_valid"]     is True
    assert r5["format_penalty"] == 0.1
    print(f"[5] Markdown fence — penalty={r5['format_penalty']} ✓")

    # 6. Wrong value
    wrong_val = '{"name": "Priya Nair", "age": 21, "city": "Chennai"}'
    r6 = evaluate_json_response(wrong_val, expected, "json_001")
    assert r6["field_precision"] < 1.0
    print(f"[6] Wrong value — precision={r6['field_precision']} ✓")

    # 7. Batch evaluator
    log_records = [
        {
            "task_type":     "json_task",
            "record_id":     "json_001",
            "response_text": perfect,
            "ground_truth":  expected,
            "success":       True,
            "run_id":        "test_run",
            "model_id":      "model_3b",
            "strategy":      "modular",
            "noise_level":   "short",
            "temperature":   0.0,
            "repeat_index":  1,
            "latency_ms":    1200,
            "error":         "",
        },
        {
            "task_type":     "json_task",
            "record_id":     "json_002",
            "response_text": "",
            "ground_truth":  expected,
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
    batch = evaluate_json_batch(log_records)
    assert len(batch) == 2
    assert batch[0]["final_score"] == 1.0
    assert batch[1]["final_score"] == 0.0
    print(f"[7] Batch evaluator — scores={[b['final_score'] for b in batch]} ✓")

    print("\n✅  All json_eval.py tests passed.")