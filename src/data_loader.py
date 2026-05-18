"""
data_loader.py
--------------
Loads, validates, and samples datasets for experiment runs.

Responsibilities:
  - Load raw and processed JSON datasets from disk
  - Validate that every record has required fields
  - Return clean Python list of dicts ready for experiment consumption
  - Support debug mode (small sample for quick testing)
  - Filter by difficulty or subtype if needed

Rules:
  - Raises explicit errors with actionable messages
  - Never silently drops records — always reports what was skipped and why
  - Imports only from utils.py and stdlib/third-party
"""

import random
from typing import Optional

from utils import load_yaml, read_json, set_seed


# ─────────────────────────────────────────────────────────────
# REQUIRED FIELDS PER TASK TYPE
# ─────────────────────────────────────────────────────────────

# Every record must have these keys at minimum.
REQUIRED_FIELDS = {
    "json_task": {
        "record":   ["id", "input", "expected_output", "metadata"],
        "metadata": ["fields_present", "difficulty", "subtype"],
        "expected_output_type": dict,
    },
    "struct_task": {
        "record":   ["id", "input", "reference_output", "metadata"],
        "metadata": ["type", "scenario", "tone", "sections_required"],
        "expected_output_type": str,
    },
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_STRUCT_TYPES = {"email", "report"}


# ─────────────────────────────────────────────────────────────
# CORE LOADER
# ─────────────────────────────────────────────────────────────

def _validate_record(record: dict, task_type: str, index: int) -> list[str]:
    """
    Validates a single record against the schema for its task type.

    Args:
        record:    The record dict to validate.
        task_type: 'json_task' or 'struct_task'
        index:     Position in the dataset list (for error messages).

    Returns:
        List of error strings. Empty list means valid.
    """
    errors = []
    schema = REQUIRED_FIELDS[task_type]

    # Check top-level required fields
    for field in schema["record"]:
        if field not in record:
            errors.append(f"  Record[{index}] (id={record.get('id', '?')}): missing field '{field}'")

    if errors:
        return errors  # can't validate deeper without base fields

    # Check metadata fields
    meta = record.get("metadata", {})
    for field in schema["metadata"]:
        if field not in meta:
            errors.append(
                f"  Record[{index}] (id={record['id']}): missing metadata field '{field}'"
            )

    # Check id is non-empty string
    if not isinstance(record["id"], str) or not record["id"].strip():
        errors.append(f"  Record[{index}]: 'id' must be a non-empty string")

    # Check input is non-empty string
    if not isinstance(record["input"], str) or not record["input"].strip():
        errors.append(f"  Record[{index}] (id={record['id']}): 'input' must be a non-empty string")

    # Check output type
    output_key = "expected_output" if task_type == "json_task" else "reference_output"
    expected_type = schema["expected_output_type"]
    if not isinstance(record.get(output_key), expected_type):
        errors.append(
            f"  Record[{index}] (id={record['id']}): '{output_key}' must be "
            f"{expected_type.__name__}, got {type(record.get(output_key)).__name__}"
        )

    # Task-specific checks
    if task_type == "json_task":
        diff = meta.get("difficulty", "")
        if diff not in VALID_DIFFICULTIES:
            errors.append(
                f"  Record[{index}] (id={record['id']}): "
                f"difficulty '{diff}' not in {VALID_DIFFICULTIES}"
            )

    if task_type == "struct_task":
        stype = meta.get("type", "")
        if stype not in VALID_STRUCT_TYPES:
            errors.append(
                f"  Record[{index}] (id={record['id']}): "
                f"type '{stype}' not in {VALID_STRUCT_TYPES}"
            )

    return errors


def load_dataset(
    task_type: str,
    split: str = "processed",
    strict: bool = True,
) -> list[dict]:
    """
    Loads a dataset for the given task type and split.

    Args:
        task_type: 'json_task' or 'struct_task'
        split:     'raw' or 'processed' (default: 'processed')
                   'raw'       → data/raw/json_task.json
                   'processed' → data/processed/json_clean.json
        strict:    If True (default), raises an error if any record fails validation.
                   If False, skips invalid records with a warning.

    Returns:
        List of valid record dicts.

    Raises:
        ValueError: For unknown task_type or split, or validation errors in strict mode.
        FileNotFoundError: If the dataset file does not exist.
    """
    # ── resolve file path ──
    if task_type not in REQUIRED_FIELDS:
        raise ValueError(
            f"[data_loader] Unknown task_type: '{task_type}'. "
            f"Must be one of: {list(REQUIRED_FIELDS.keys())}"
        )

    if split == "raw":
        filename = "json_task.json" if task_type == "json_task" else "structured_task.json"
        path = f"data/raw/{filename}"
    elif split == "processed":
        filename = "json_clean.json" if task_type == "json_task" else "structured_clean.json"
        path = f"data/processed/{filename}"
    else:
        raise ValueError(
            f"[data_loader] Unknown split: '{split}'. Must be 'raw' or 'processed'."
        )

    # ── load ──
    data = read_json(path)

    if not isinstance(data, list):
        raise ValueError(
            f"[data_loader] Dataset must be a JSON array (list), "
            f"got {type(data).__name__}: {path}"
        )

    if len(data) == 0:
        raise ValueError(f"[data_loader] Dataset is empty: {path}")

    # ── validate ──
    all_errors = []
    valid_records = []

    for i, record in enumerate(data):
        errors = _validate_record(record, task_type, i)
        if errors:
            all_errors.extend(errors)
            if not strict:
                print(f"[data_loader] WARNING: Skipping invalid record at index {i}: {record.get('id', '?')}")
        else:
            valid_records.append(record)

    if all_errors and strict:
        error_summary = "\n".join(all_errors)
        raise ValueError(
            f"[data_loader] Validation failed for '{path}'.\n"
            f"  {len(all_errors)} error(s) found:\n{error_summary}\n\n"
            f"  Fix the dataset or use strict=False to skip invalid records."
        )

    if all_errors and not strict:
        print(
            f"[data_loader] WARNING: {len(all_errors)} validation error(s) found. "
            f"{len(valid_records)} valid records loaded."
        )

    print(
        f"[data_loader] Loaded {len(valid_records)} records "
        f"from '{path}' (task={task_type}, split={split})"
    )

    return valid_records


# ─────────────────────────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────────────────────────

def sample_dataset(
    records: list[dict],
    n: int,
    seed: int = 42,
    difficulty: Optional[str] = None,
    subtype: Optional[str] = None,
) -> list[dict]:
    """
    Returns a random sample of n records from a dataset.
    Optionally filters by difficulty or subtype before sampling.

    Args:
        records:    Full dataset list from load_dataset().
        n:          Number of records to sample.
        seed:       Random seed for reproducibility.
        difficulty: If set, filter to only records with this difficulty
                    ('easy', 'medium', 'hard'). JSON task only.
        subtype:    If set, filter to only records with this subtype.

    Returns:
        List of sampled record dicts.

    Raises:
        ValueError: If n > available records after filtering.
    """
    filtered = records

    if difficulty is not None:
        filtered = [r for r in filtered if r["metadata"].get("difficulty") == difficulty]
        if not filtered:
            raise ValueError(
                f"[data_loader] No records found with difficulty='{difficulty}'. "
                f"Valid values: {VALID_DIFFICULTIES}"
            )

    if subtype is not None:
        filtered = [r for r in filtered if r["metadata"].get("subtype") == subtype]
        if not filtered:
            raise ValueError(
                f"[data_loader] No records found with subtype='{subtype}'."
            )

    if n > len(filtered):
        raise ValueError(
            f"[data_loader] Requested {n} samples but only {len(filtered)} "
            f"records available after filtering."
        )

    set_seed(seed)
    return random.sample(filtered, n)


def load_debug_samples(n: int = 5) -> dict:
    """
    Loads a small sample of records for quick pipeline testing.
    Uses data/samples/debug_samples.json if it exists,
    otherwise samples from the raw datasets.

    Args:
        n: Number of samples per task type.

    Returns:
        {
            'json_task':   [n records],
            'struct_task': [n records],
        }
    """
    try:
        debug_data = read_json("data/samples/debug_samples.json")
        # If the debug file has content, return it
        if isinstance(debug_data, list) and len(debug_data) > 0:
            print(f"[data_loader] Loaded {len(debug_data)} debug samples from debug_samples.json")
            return {"debug": debug_data}
    except (FileNotFoundError, Exception):
        pass  # Fall back to sampling from raw

    # Sample from raw datasets
    print("[data_loader] debug_samples.json empty — sampling from raw datasets")
    jt = load_dataset("json_task", split="raw")
    st = load_dataset("struct_task", split="raw")

    return {
        "json_task":   sample_dataset(jt, min(n, len(jt))),
        "struct_task": sample_dataset(st, min(n, len(st))),
    }


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

def dataset_stats(records: list[dict], task_type: str) -> dict:
    """
    Returns summary statistics about a loaded dataset.

    Args:
        records:   List of records from load_dataset().
        task_type: 'json_task' or 'struct_task'

    Returns:
        Dict of summary statistics.
    """
    stats = {"total": len(records), "task_type": task_type}

    if task_type == "json_task":
        difficulties = {}
        subtypes = {}
        fields_seen = {}
        for r in records:
            meta = r["metadata"]
            d = meta.get("difficulty", "unknown")
            s = meta.get("subtype", "unknown")
            difficulties[d] = difficulties.get(d, 0) + 1
            subtypes[s] = subtypes.get(s, 0) + 1
            for f in meta.get("fields_present", []):
                fields_seen[f] = fields_seen.get(f, 0) + 1

        stats["difficulty_distribution"] = difficulties
        stats["unique_subtypes"] = len(subtypes)
        stats["field_coverage"] = fields_seen

    elif task_type == "struct_task":
        types = {}
        tones = {}
        for r in records:
            meta = r["metadata"]
            t = meta.get("type", "unknown")
            tn = meta.get("tone", "unknown")
            types[t] = types.get(t, 0) + 1
            tones[tn] = tones.get(tn, 0) + 1

        stats["type_distribution"] = types
        stats["unique_tones"] = len(tones)
        stats["unique_scenarios"] = len({r["metadata"].get("scenario") for r in records})

    return stats


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run directly: python src/data_loader.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== data_loader.py self-test ===\n")

    # 1. Load JSON task from raw
    jt = load_dataset("json_task", split="raw")
    assert len(jt) == 150, f"Expected 150 JSON task records, got {len(jt)}"
    print(f"[1] JSON task loaded: {len(jt)} records ✓")

    # 2. Load structured task from raw
    st = load_dataset("struct_task", split="raw")
    assert len(st) == 150, f"Expected 150 struct task records, got {len(st)}"
    print(f"[2] Structured task loaded: {len(st)} records ✓")

    # 3. Stats
    jt_stats = dataset_stats(jt, "json_task")
    print(f"[3] JSON task stats: {jt_stats}")
    assert jt_stats["total"] == 150

    st_stats = dataset_stats(st, "struct_task")
    print(f"[4] Struct task stats: {st_stats}")
    assert st_stats["total"] == 150

    # 4. Sampling
    sample = sample_dataset(jt, n=10, seed=42)
    assert len(sample) == 10
    ids = [r["id"] for r in sample]
    print(f"[5] Sample of 10 IDs: {ids} ✓")

    # Reproducibility check
    sample2 = sample_dataset(jt, n=10, seed=42)
    assert [r["id"] for r in sample2] == ids, "Sampling not reproducible!"
    print(f"[6] Sampling reproducibility OK ✓")

    # 5. Filter by difficulty
    easy = sample_dataset(jt, n=5, seed=42, difficulty="easy")
    assert all(r["metadata"]["difficulty"] == "easy" for r in easy)
    print(f"[7] Difficulty filter OK: {[r['id'] for r in easy]} ✓")

    # 6. Invalid task type
    try:
        load_dataset("bad_task")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"[8] ValueError on bad task_type: OK ✓")

    # 7. Load debug samples
    debug = load_debug_samples(n=3)
    assert "json_task" in debug or "debug" in debug
    print(f"[9] Debug samples loaded ✓")

    print("\n✅  All data_loader.py tests passed.")
