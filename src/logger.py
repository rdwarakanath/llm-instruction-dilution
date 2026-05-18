"""
logger.py
---------
Structured JSON logging for every experiment run.

Every single model call in the experiment is logged as one JSON record.
This is critical for:
  - Reproducibility (know exactly what was sent and received)
  - Debugging (find failures without re-running)
  - Analysis (evaluation/metrics.py reads these logs)
  - Paper (raw logs are a methodology artifact)

Log format (one record per model call):
  {
    "run_id":          str,
    "record_id":       str,
    "model_id":        str,
    "model_name":      str,
    "strategy":        str,
    "task_type":       str,
    "noise_level":     str,
    "temperature":     float,
    "attempt_number":  int,
    "system_prompt":   str,
    "user_prompt":     str,
    "response_text":   str,
    "success":         bool,
    "error":           str,
    "latency_ms":      int,
    "token_counts":    {system: int, user: int, total: int},
    "budget_ok":       bool,
    "timestamp":       str,   # UTC ISO format
  }

Two log destinations:
  1. results/logs/run_logs.jsonl  — append-only, one record per line
  2. results/raw_outputs/<model_id>/<run_id>_<record_id>.json — one file per call

Rules:
  - Logging must NEVER raise an exception and crash the experiment.
    All logger functions catch their own errors and print warnings.
  - Imports only from utils.py and stdlib
"""

import traceback
from datetime import datetime, timezone
from typing import Optional

from utils import append_json_line, get_project_root, write_json


# ─────────────────────────────────────────────────────────────
# RECORD BUILDER
# ─────────────────────────────────────────────────────────────

def build_log_record(
    run_id: str,
    record_id: str,
    model_id: str,
    model_name: str,
    strategy: str,
    task_type: str,
    noise_level: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    success: bool,
    error: str,
    latency_ms: int,
    token_counts: dict,
    budget_ok: bool,
    attempt_number: int = 1,
    repeat_index: int = 1,
    ground_truth: Optional[str] = None,
) -> dict:
    """
    Builds a complete, structured log record for one model call.

    Args:
        run_id:         Unique run identifier, e.g. 'exp_prompt_length_20240315_143022'
        record_id:      Dataset record ID, e.g. 'json_001'
        model_id:       Config model ID, e.g. 'model_3b'
        model_name:     Ollama model name, e.g. 'llama3.2:3b'
        strategy:       Prompt strategy, e.g. 'modular'
        task_type:      'json_task' or 'struct_task'
        noise_level:    'short', 'medium', or 'long'
        temperature:    Sampling temperature used
        system_prompt:  The system prompt text sent to the model
        user_prompt:    The user prompt text sent to the model
        response_text:  Raw model output
        success:        Whether the model call succeeded
        error:          Error message (empty string if success)
        latency_ms:     Response time in milliseconds
        token_counts:   {'system': int, 'user': int, 'total': int}
        budget_ok:      Whether the prompt was within token budget
        attempt_number: Which retry attempt produced this response (1-based)
        repeat_index:   Which repeat of this cell this is (for stability scoring)
        ground_truth:   Expected output from dataset (for evaluation reference)

    Returns:
        Complete log record dict.
    """
    return {
        "run_id":          run_id,
        "record_id":       record_id,
        "model_id":        model_id,
        "model_name":      model_name,
        "strategy":        strategy,
        "task_type":       task_type,
        "noise_level":     noise_level,
        "temperature":     temperature,
        "repeat_index":    repeat_index,
        "attempt_number":  attempt_number,
        "system_prompt":   system_prompt,
        "user_prompt":     user_prompt,
        "response_text":   response_text,
        "success":         success,
        "error":           error,
        "latency_ms":      latency_ms,
        "token_counts":    token_counts,
        "budget_ok":       budget_ok,
        "ground_truth":    ground_truth,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# LOG FROM RUNNER OUTPUT
# ─────────────────────────────────────────────────────────────

def log_from_runner_result(
    runner_result: dict,
    prompt: dict,
    run_id: str,
    repeat_index: int = 1,
    ground_truth: Optional[str] = None,
    model_name: str = "",
) -> dict:
    """
    Convenience function: builds and saves a log record from the output
    of model_runner.run_from_prompt_dict().

    Args:
        runner_result: Dict returned by model_runner.run_from_prompt_dict()
        prompt:        The prompt dict from prompt_builder.build_prompt()
        run_id:        Current experiment run ID
        repeat_index:  Repeat number for this cell (1, 2, or 3)
        ground_truth:  Expected output string for evaluation reference
        model_name:    Ollama model name (looked up from runner_result if blank)

    Returns:
        The complete log record dict (also saved to disk).
    """
    mn = model_name or runner_result.get("model_name", "")

    record = build_log_record(
        run_id         = run_id,
        record_id      = runner_result.get("record_id", ""),
        model_id       = runner_result.get("model_id", ""),
        model_name     = mn,
        strategy       = runner_result.get("strategy", ""),
        task_type      = runner_result.get("task_type", ""),
        noise_level    = runner_result.get("noise_level", ""),
        temperature    = runner_result.get("temperature", 0.0),
        system_prompt  = prompt.get("system", ""),
        user_prompt    = prompt.get("user", ""),
        response_text  = runner_result.get("response_text", ""),
        success        = runner_result.get("success", False),
        error          = runner_result.get("error", ""),
        latency_ms     = runner_result.get("latency_ms", 0),
        token_counts   = runner_result.get("token_counts", {}),
        budget_ok      = prompt.get("budget_ok", True),
        attempt_number = runner_result.get("attempt", 1),
        repeat_index   = repeat_index,
        ground_truth   = ground_truth,
    )

    save_log_record(record, run_id)
    return record


# ─────────────────────────────────────────────────────────────
# SAVE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def save_log_record(record: dict, run_id: str) -> None:
    """
    Saves a log record to two destinations:
      1. results/logs/run_logs.jsonl        (append one line)
      2. results/raw_outputs/<model_id>/    (one JSON file per call)

    Silently catches all errors to avoid crashing the experiment.

    Args:
        record: Complete log record dict from build_log_record().
        run_id: Current run ID (used for the raw output filename).
    """
    # ── destination 1: central JSONL log ──
    try:
        append_json_line(record, "results/logs/run_logs.jsonl")
    except Exception as e:
        print(f"[logger] WARNING: Could not write to run_logs.jsonl: {e}")

    # ── destination 2: per-call raw output file ──
    try:
        model_id  = record.get("model_id", "unknown")
        record_id = record.get("record_id", "unknown")
        strategy  = record.get("strategy", "")
        noise     = record.get("noise_level", "")
        repeat    = record.get("repeat_index", 1)
        temp_str  = str(record.get("temperature", 0.0)).replace(".", "p")

        filename = f"{run_id}_{record_id}_{strategy}_{noise}_t{temp_str}_r{repeat}.json"
        path     = f"results/raw_outputs/{model_id}/{filename}"

        write_json(record, path, overwrite=True)
    except Exception as e:
        print(f"[logger] WARNING: Could not write raw output file: {e}")


def save_run_summary(
    run_id: str,
    total_calls: int,
    successful_calls: int,
    failed_calls: int,
    config_snapshot: dict,
    duration_seconds: float,
) -> None:
    """
    Saves a summary JSON file for the entire run.
    Stored at: results/metadata/<run_id>_summary.json

    Args:
        run_id:           Run identifier.
        total_calls:      Total model calls made.
        successful_calls: How many succeeded.
        failed_calls:     How many failed.
        config_snapshot:  The experiment config used for this run.
        duration_seconds: Total run duration in seconds.
    """
    summary = {
        "run_id":           run_id,
        "total_calls":      total_calls,
        "successful_calls": successful_calls,
        "failed_calls":     failed_calls,
        "success_rate":     round(successful_calls / total_calls, 4) if total_calls > 0 else 0.0,
        "duration_seconds": round(duration_seconds, 2),
        "config_snapshot":  config_snapshot,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    try:
        path = f"results/metadata/{run_id}_summary.json"
        write_json(summary, path, overwrite=True)
        print(f"[logger] Run summary saved: {path}")
    except Exception as e:
        print(f"[logger] WARNING: Could not save run summary: {e}")


# ─────────────────────────────────────────────────────────────
# LOG READER  (for evaluation stage)
# ─────────────────────────────────────────────────────────────

def load_run_logs(run_id: Optional[str] = None) -> list[dict]:
    """
    Loads all log records from results/logs/run_logs.jsonl.
    Optionally filters to a specific run_id.

    Args:
        run_id: If provided, returns only records matching this run_id.

    Returns:
        List of log record dicts.

    Raises:
        FileNotFoundError: If run_logs.jsonl does not exist.
    """
    import json as _json

    log_path = get_project_root() / "results" / "logs" / "run_logs.jsonl"

    if not log_path.exists():
        raise FileNotFoundError(
            f"[logger] Log file not found: {log_path}\n"
            f"  No experiments have been run yet."
        )

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = _json.loads(line)
                if run_id is None or record.get("run_id") == run_id:
                    records.append(record)
            except _json.JSONDecodeError as e:
                print(f"[logger] WARNING: Skipping malformed JSON on line {line_num}: {e}")

    return records


def get_run_ids() -> list[str]:
    """
    Returns a sorted list of all unique run_ids in the log file.
    """
    try:
        records = load_run_logs()
        return sorted(set(r["run_id"] for r in records))
    except FileNotFoundError:
        return []


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run directly: python src/logger.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== logger.py self-test ===\n")

    TEST_RUN_ID = "test_run_20240315_000000"

    # 1. Build a log record
    record = build_log_record(
        run_id         = TEST_RUN_ID,
        record_id      = "json_001",
        model_id       = "model_3b",
        model_name     = "llama3.2:3b",
        strategy       = "modular",
        task_type      = "json_task",
        noise_level    = "short",
        temperature    = 0.0,
        system_prompt  = "",
        user_prompt    = "Ravi Kumar is 21 years old.",
        response_text  = '{"name": "Ravi Kumar", "age": 21}',
        success        = True,
        error          = "",
        latency_ms     = 1420,
        token_counts   = {"system": 0, "user": 15, "total": 15},
        budget_ok      = True,
        attempt_number = 1,
        repeat_index   = 1,
        ground_truth   = '{"name": "Ravi Kumar", "age": 21}',
    )

    assert record["run_id"]         == TEST_RUN_ID
    assert record["success"]        is True
    assert record["token_counts"]   == {"system": 0, "user": 15, "total": 15}
    assert "timestamp" in record
    print(f"[1] build_log_record OK ✓")

    # 2. Save the record
    save_log_record(record, TEST_RUN_ID)
    print(f"[2] save_log_record OK ✓")

    # 3. Read it back from jsonl
    logs = load_run_logs(run_id=TEST_RUN_ID)
    assert len(logs) >= 1
    assert logs[-1]["record_id"] == "json_001"
    print(f"[3] load_run_logs OK: {len(logs)} record(s) for run_id='{TEST_RUN_ID}' ✓")

    # 4. get_run_ids
    ids = get_run_ids()
    assert TEST_RUN_ID in ids
    print(f"[4] get_run_ids OK: {ids} ✓")

    # 5. save_run_summary
    save_run_summary(
        run_id           = TEST_RUN_ID,
        total_calls      = 1,
        successful_calls = 1,
        failed_calls     = 0,
        config_snapshot  = {"test": True},
        duration_seconds = 1.42,
    )
    print(f"[5] save_run_summary OK ✓")

    # 6. Failed record (ensure it logs without crashing)
    failed_record = build_log_record(
        run_id         = TEST_RUN_ID,
        record_id      = "json_002",
        model_id       = "model_3b",
        model_name     = "llama3.2:3b",
        strategy       = "monolithic",
        task_type      = "json_task",
        noise_level    = "long",
        temperature    = 0.7,
        system_prompt  = "",
        user_prompt    = "Some input.",
        response_text  = "",
        success        = False,
        error          = "Connection timed out.",
        latency_ms     = 0,
        token_counts   = {"system": 0, "user": 5, "total": 5},
        budget_ok      = True,
        attempt_number = 3,
        repeat_index   = 2,
    )
    save_log_record(failed_record, TEST_RUN_ID)
    print(f"[6] Failed record logging OK ✓")

    # 7. Check log has both records
    logs_after = load_run_logs(run_id=TEST_RUN_ID)
    record_ids = [r["record_id"] for r in logs_after]
    assert "json_001" in record_ids
    assert "json_002" in record_ids
    print(f"[7] Both records in log ✓")

    print("\n✅  All logger.py tests passed.")
