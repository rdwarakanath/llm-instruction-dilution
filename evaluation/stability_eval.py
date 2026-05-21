"""
stability_eval.py
-----------------
Measures output consistency (stability) across repeated runs
of the same experimental cell.

Each cell in the experiment matrix is run 3 times (repeat_index 1,2,3).
This module groups those runs and measures how much the scores vary.

Why this matters for the paper:
  - A strategy might score well on average but be highly unstable
  - Stability is a separate research dimension from accuracy
  - Small models at high temperatures are expected to be more unstable

Stability metrics:
  - score_std       : standard deviation of final_score across repeats
  - score_range     : max - min final_score
  - score_mean      : mean final_score
  - consistency     : 1.0 - score_std  (higher = more consistent)
  - response_unique : number of unique response texts (1 = perfectly stable)

A cell is identified by: (run_id, model_id, strategy, noise_level, temperature, record_id)

Rules:
  - Imports only from metrics.py and stdlib
  - Reads evaluation results, not raw log records
"""

import hashlib
import itertools
from collections import defaultdict
from typing import Optional

from metrics import response_variance


# ─────────────────────────────────────────────────────────────
# CELL KEY
# ─────────────────────────────────────────────────────────────

def _cell_key(record: dict) -> tuple:
    """
    Returns a tuple that uniquely identifies an experimental cell
    (ignoring repeat_index).

    Args:
        record: An evaluation result dict (from json_eval or structured_eval).

    Returns:
        Tuple of (run_id, model_id, strategy, noise_level, temperature, record_id)
    """
    return (
        record.get("run_id", ""),
        record.get("model_id", ""),
        record.get("strategy", ""),
        record.get("noise_level", ""),
        str(record.get("temperature", 0.0)),
        record.get("record_id", ""),
    )


# ─────────────────────────────────────────────────────────────
# CORE STABILITY EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_cell_stability(eval_records: list) -> dict:
    """
    Computes stability metrics for a group of evaluation records
    that all belong to the same experimental cell (same parameters,
    different repeat_index values).

    Args:
        eval_records: List of evaluation result dicts for one cell.
                      Each must have 'final_score' and optionally
                      'response_text' from the original log.

    Returns:
        {
            'n_repeats':        int,
            'score_mean':       float,
            'score_std':        float,
            'score_var':        float,
            'score_min':        float,
            'score_max':        float,
            'score_range':      float,
            'consistency':      float,   # 1.0 - score_std (clamped to [0,1])
            'all_scores':       list,
            'cell_key':         tuple,
        }
    """
    if not eval_records:
        raise ValueError("[stability_eval] eval_records list is empty.")

    scores = [r["final_score"] for r in eval_records]

    if len(scores) == 1:
        # Only one repeat — can't compute variance, return defaults
        return {
            "n_repeats":   1,
            "score_mean":  round(scores[0], 4),
            "score_std":   0.0,
            "score_var":   0.0,
            "score_min":   round(scores[0], 4),
            "score_max":   round(scores[0], 4),
            "score_range": 0.0,
            "consistency": 1.0,
            "all_scores":  scores,
            "cell_key":    _cell_key(eval_records[0]),
        }

    var_stats = response_variance(scores)

    consistency = max(0.0, min(1.0, 1.0 - var_stats["std"]))

    return {
        "n_repeats":   var_stats["n"],
        "score_mean":  var_stats["mean"],
        "score_std":   var_stats["std"],
        "score_var":   var_stats["var"],
        "score_min":   var_stats["min"],
        "score_max":   var_stats["max"],
        "score_range": var_stats["range"],
        "consistency": round(consistency, 4),
        "all_scores":  scores,
        "cell_key":    _cell_key(eval_records[0]),
    }


# ─────────────────────────────────────────────────────────────
# BATCH STABILITY (groups all cells automatically)
# ─────────────────────────────────────────────────────────────

def evaluate_all_stability(
    eval_results: list,
    log_records: Optional[list] = None,
) -> list:
    """
    Groups evaluation results by experimental cell and computes
    stability metrics for every cell.

    Args:
        eval_results: List of evaluation result dicts from
                      json_eval.evaluate_json_batch() or
                      structured_eval.evaluate_struct_batch().
        log_records:  Optional list of raw log records (from logger).
                      If provided, unique response text count is computed.

    Returns:
        List of stability result dicts, one per unique cell.
        Each dict includes the cell metadata (model_id, strategy, etc.)
        plus all stability metrics.
    """
    # Group by cell key
    cells = defaultdict(list)
    for r in eval_results:
        key = _cell_key(r)
        cells[key].append(r)

    # Build a lookup of log records by cell key + repeat_index (for response text)
    log_lookup = {}
    if log_records:
        for lr in log_records:
            lk = (
                lr.get("run_id", ""),
                lr.get("model_id", ""),
                lr.get("strategy", ""),
                lr.get("noise_level", ""),
                str(lr.get("temperature", 0.0)),
                lr.get("record_id", ""),
            )
            ri = lr.get("repeat_index", 1)
            log_lookup[(lk, ri)] = lr.get("response_text", "")

    stability_results = []

    for key, records in cells.items():
        stability = evaluate_cell_stability(records)

        # Count unique responses if log_records available
        if log_records:
            response_texts = set()
            for r in records:
                ri = r.get("repeat_index", 1)
                rt = log_lookup.get((key, ri), "")
                # Hash to avoid storing huge strings
                response_texts.add(hashlib.md5(rt.encode()).hexdigest())
            stability["response_unique"] = len(response_texts)
        else:
            stability["response_unique"] = None

        # Add cell metadata from first record
        first = records[0]
        stability["run_id"]      = first.get("run_id", "")
        stability["model_id"]    = first.get("model_id", "")
        stability["strategy"]    = first.get("strategy", "")
        stability["noise_level"] = first.get("noise_level", "")
        stability["temperature"] = first.get("temperature", 0.0)
        stability["record_id"]   = first.get("record_id", "")
        stability["task_type"]   = first.get("task_type",
                                   "json_task" if "field_f1" in first else "struct_task")

        stability_results.append(stability)

    return stability_results


# ─────────────────────────────────────────────────────────────
# AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────

def aggregate_stability_by_condition(
    stability_results: list,
    group_by: list = None,
) -> list:
    """
    Aggregates cell-level stability results into condition-level summaries.

    Useful for the paper: "modular at medium noise has mean consistency X".

    Args:
        stability_results: Output of evaluate_all_stability().
        group_by: List of keys to group by.
                  Default: ['model_id', 'strategy', 'noise_level']

    Returns:
        List of aggregated dicts, one per group.
        Each has: group keys + mean/std of score_mean, consistency, score_std.
    """
    if group_by is None:
        group_by = ["model_id", "strategy", "noise_level"]

    groups = defaultdict(list)
    for r in stability_results:
        key = tuple(r.get(k, "") for k in group_by)
        groups[key].append(r)

    aggregated = []
    for key_vals, group_records in groups.items():
        group_dict = dict(zip(group_by, key_vals))

        score_means   = [r["score_mean"]  for r in group_records]
        consistencies = [r["consistency"] for r in group_records]
        score_stds    = [r["score_std"]   for r in group_records]

        n = len(score_means)
        group_dict["n_cells"]          = n
        group_dict["avg_score_mean"]   = round(sum(score_means) / n, 4)
        group_dict["avg_consistency"]  = round(sum(consistencies) / n, 4)
        group_dict["avg_score_std"]    = round(sum(score_stds) / n, 4)
        group_dict["avg_score_range"]  = round(
            sum(r["score_range"] for r in group_records) / n, 4
        )

        aggregated.append(group_dict)

    # Sort for readability
    aggregated.sort(key=lambda x: (
        x.get("model_id", ""),
        x.get("strategy", ""),
        x.get("noise_level", ""),
    ))

    return aggregated


def worst_cells(stability_results: list, n: int = 10) -> list:
    """
    Returns the n cells with the lowest consistency (most unstable).
    Useful for debugging and paper discussion section.

    Args:
        stability_results: Output of evaluate_all_stability().
        n:                 Number of worst cells to return.

    Returns:
        List of n stability dicts sorted by consistency ascending.
    """
    return sorted(stability_results, key=lambda x: x["consistency"])[:n]


def best_cells(stability_results: list, n: int = 10) -> list:
    """
    Returns the n cells with the highest consistency (most stable).

    Args:
        stability_results: Output of evaluate_all_stability().
        n:                 Number of best cells to return.

    Returns:
        List of n stability dicts sorted by consistency descending.
    """
    return sorted(stability_results, key=lambda x: x["consistency"], reverse=True)[:n]


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python src/stability_eval.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== stability_eval.py self-test ===\n")

    # Mock evaluation results — 3 repeats of the same cell
    def _mock_eval(record_id, strategy, noise, model, temp, repeat, score):
        return {
            "run_id":      "test_run",
            "record_id":   record_id,
            "model_id":    model,
            "strategy":    strategy,
            "noise_level": noise,
            "temperature": temp,
            "repeat_index": repeat,
            "final_score": score,
            "field_f1":    score,  # marks it as json_task
        }

    # Cell A: very stable (low variance)
    cell_a = [
        _mock_eval("json_001", "modular", "short", "model_3b", 0.0, 1, 0.92),
        _mock_eval("json_001", "modular", "short", "model_3b", 0.0, 2, 0.91),
        _mock_eval("json_001", "modular", "short", "model_3b", 0.0, 3, 0.93),
    ]

    # Cell B: unstable (high variance)
    cell_b = [
        _mock_eval("json_001", "monolithic", "long", "model_3b", 1.0, 1, 0.95),
        _mock_eval("json_001", "monolithic", "long", "model_3b", 1.0, 2, 0.40),
        _mock_eval("json_001", "monolithic", "long", "model_3b", 1.0, 3, 0.70),
    ]

    # 1. Stable cell
    s_a = evaluate_cell_stability(cell_a)
    assert s_a["n_repeats"]  == 3
    assert s_a["score_std"]  < 0.02
    assert s_a["consistency"] > 0.98
    print(f"[1] Stable cell — std={s_a['score_std']}, consistency={s_a['consistency']} ✓")

    # 2. Unstable cell
    s_b = evaluate_cell_stability(cell_b)
    assert s_b["score_std"]   > 0.20
    assert s_b["consistency"] < 0.80
    assert s_b["score_range"] > 0.40
    print(f"[2] Unstable cell — std={s_b['score_std']}, consistency={s_b['consistency']}, range={s_b['score_range']} ✓")

    # 3. Single repeat (no variance)
    s_single = evaluate_cell_stability([cell_a[0]])
    assert s_single["score_std"]   == 0.0
    assert s_single["consistency"] == 1.0
    print(f"[3] Single repeat — std={s_single['score_std']}, consistency={s_single['consistency']} ✓")

    # 4. evaluate_all_stability (groups cells automatically)
    all_evals = cell_a + cell_b
    stability = evaluate_all_stability(all_evals)
    assert len(stability) == 2, f"Expected 2 cells, got {len(stability)}"
    print(f"[4] evaluate_all_stability — {len(stability)} cells found ✓")

    # 5. Aggregate by condition
    agg = aggregate_stability_by_condition(stability, group_by=["model_id", "strategy"])
    assert len(agg) == 2  # modular and monolithic
    for a in agg:
        print(f"    [{a['model_id']} | {a['strategy']:12s}] "
              f"avg_score={a['avg_score_mean']:.3f}  "
              f"avg_consistency={a['avg_consistency']:.3f}")
    print(f"[5] aggregate_stability_by_condition OK ✓")

    # 6. worst/best cells
    w = worst_cells(stability, n=1)
    b = best_cells(stability, n=1)
    assert w[0]["strategy"] == "monolithic"
    assert b[0]["strategy"] == "modular"
    print(f"[6] worst_cells={w[0]['strategy']}, best_cells={b[0]['strategy']} ✓")

    # 7. Empty list raises
    try:
        evaluate_cell_stability([])
        assert False
    except ValueError:
        print("[7] Empty list raises ValueError ✓")

    # 8. Aggregate with noise_level grouping
    agg2 = aggregate_stability_by_condition(
        stability, group_by=["strategy", "noise_level"]
    )
    assert all("strategy" in a for a in agg2)
    print(f"[8] Aggregate by strategy+noise_level — {len(agg2)} groups ✓")

    print("\n✅  All stability_eval.py tests passed.")
