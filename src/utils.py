"""
utils.py
--------
Shared helpers used across every module in this project.

Responsibilities:
  - Load YAML config files
  - Set global random seed (Python + NumPy)
  - Safe JSON read / write
  - Path resolution relative to project root
  - Timestamp generation for run IDs

Rules:
  - No circular imports — this module imports NOTHING from src/
  - All functions are pure (no side effects except file I/O)
  - Every function has explicit error messages
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root.

    Assumes this file lives at:  <project_root>/src/utils.py
    So project root is two levels up from this file.
    """
    return Path(__file__).resolve().parent.parent


def resolve_path(relative_path: str) -> Path:
    """
    Resolves a relative path string against the project root.

    Args:
        relative_path: e.g. 'config/models.yaml'

    Returns:
        Absolute Path object.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    full_path = get_project_root() / relative_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"[utils] Path not found: {full_path}\n"
            f"  Looked for: '{relative_path}' relative to project root '{get_project_root()}'"
        )
    return full_path


# ─────────────────────────────────────────────────────────────
# YAML CONFIG LOADER
# ─────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    """
    Loads a YAML file and returns it as a dict.

    Args:
        path: Relative path from project root, e.g. 'config/models.yaml'

    Returns:
        Parsed YAML as a Python dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or not valid YAML.
    """
    full_path = resolve_path(path)

    with open(full_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"[utils] YAML file is empty: {full_path}")

    if not isinstance(data, dict):
        raise ValueError(
            f"[utils] YAML file must contain a top-level mapping (dict), "
            f"got {type(data).__name__}: {full_path}"
        )

    return data


def load_all_configs() -> dict:
    """
    Loads all three config files at once and returns them as a combined dict.

    Returns:
        {
            'models':     <models.yaml contents>,
            'experiment': <experiment.yaml contents>,
            'prompt':     <prompt_config.yaml contents>
        }

    Raises:
        FileNotFoundError / ValueError: Propagated from load_yaml().
    """
    return {
        "models":     load_yaml("config/models.yaml"),
        "experiment": load_yaml("config/experiment.yaml"),
        "prompt":     load_yaml("config/prompt_config.yaml"),
    }


# ─────────────────────────────────────────────────────────────
# RANDOM SEED
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python's random module and NumPy.
    Call this once at the start of every experiment run.

    Args:
        seed: Integer seed value. Use config['experiment']['seed'] = 42.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"[utils] Seed must be a non-negative integer, got: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────
# JSON FILE I/O
# ─────────────────────────────────────────────────────────────

def read_json(path: str) -> Any:
    """
    Reads a JSON file and returns the parsed content.

    Args:
        path: Relative path from project root, e.g. 'data/processed/json_clean.json'

    Returns:
        Parsed JSON content (list or dict).

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    full_path = resolve_path(path)

    with open(full_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"[utils] Invalid JSON in file: {full_path}\n  Detail: {e.msg}",
                e.doc,
                e.pos,
            )


def write_json(data: Any, path: str, overwrite: bool = False) -> None:
    """
    Writes data to a JSON file at the given path.

    Args:
        data:      Python object (list or dict) to serialise.
        path:      Relative path from project root, e.g. 'results/logs/run_logs.json'
        overwrite: If False (default), raises an error if the file already exists.

    Raises:
        FileExistsError: If file exists and overwrite=False.
        TypeError: If data is not JSON-serialisable.
    """
    full_path = get_project_root() / path

    if full_path.exists() and not overwrite:
        raise FileExistsError(
            f"[utils] File already exists: {full_path}\n"
            f"  Pass overwrite=True to replace it."
        )

    # Ensure parent directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_json_line(record: dict, path: str) -> None:
    """
    Appends a single JSON record as a new line to a .jsonl (JSON Lines) file.
    Creates the file if it does not exist.

    Used by logger.py to stream results without loading the full file.

    Args:
        record: A dict representing one log entry.
        path:   Relative path from project root, e.g. 'results/logs/run_logs.jsonl'
    """
    if not isinstance(record, dict):
        raise TypeError(f"[utils] append_json_line expects a dict, got {type(record).__name__}")

    full_path = get_project_root() / path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# TEXT FILE I/O
# ─────────────────────────────────────────────────────────────

def read_text(path: str) -> str:
    """
    Reads a plain text file and returns its contents as a string.
    Used for loading skill blocks and noise blocks.

    Args:
        path: Relative path from project root, e.g. 'skills/json_skill.txt'

    Returns:
        File contents as a stripped string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    full_path = resolve_path(path)

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ─────────────────────────────────────────────────────────────
# TIMESTAMP / RUN ID
# ─────────────────────────────────────────────────────────────

def get_timestamp() -> str:
    """
    Returns the current UTC timestamp as a sortable string.
    Format: YYYYMMDD_HHMMSS

    Used to generate unique run IDs.

    Returns:
        e.g. '20240315_143022'
    """
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def generate_run_id(prefix: str = "run") -> str:
    """
    Generates a unique run identifier combining a prefix and timestamp.

    Args:
        prefix: Short label, e.g. 'exp_prompt_length' or 'baseline'

    Returns:
        e.g. 'exp_prompt_length_20240315_143022'
    """
    return f"{prefix}_{get_timestamp()}"


# ─────────────────────────────────────────────────────────────
# METADATA SNAPSHOT
# ─────────────────────────────────────────────────────────────

def save_run_metadata(config: dict, run_id: str) -> None:
    """
    Saves a snapshot of the experiment config used for a specific run.
    Stored in results/metadata/<run_id>_config.json for reproducibility.

    Args:
        config: The full config dict used for this run.
        run_id: Unique run identifier from generate_run_id().
    """
    path = f"results/metadata/{run_id}_config.json"
    write_json(config, path, overwrite=True)


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run directly: python src/utils.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== utils.py self-test ===\n")

    # 1. Project root
    root = get_project_root()
    print(f"[1] Project root: {root}")
    assert root.exists(), "Project root does not exist!"

    # 2. YAML loading
    cfg = load_all_configs()
    assert "models" in cfg
    assert "experiment" in cfg
    assert "prompt" in cfg
    print(f"[2] Configs loaded OK: {list(cfg.keys())}")

    # 3. Seed
    set_seed(42)
    v1 = random.random()
    set_seed(42)
    v2 = random.random()
    assert v1 == v2, "Seed not reproducible!"
    print(f"[3] Seed reproducibility OK: {v1:.6f}")

    # 4. Timestamp / run ID
    ts = get_timestamp()
    rid = generate_run_id("test")
    assert rid.startswith("test_"), f"Run ID malformed: {rid}"
    print(f"[4] Run ID: {rid}")

    # 5. read_text — verify skills file exists
    try:
        txt = read_text("skills/json_skill.txt")
        print(f"[5] read_text OK: {len(txt)} chars from json_skill.txt")
    except FileNotFoundError as e:
        print(f"[5] SKIP (skills file not present yet): {e}")

    # 6. write_json + read_json round-trip
    import tempfile, shutil
    test_data = {"test": True, "values": [1, 2, 3]}
    tmp_path = "results/logs/_test_utils.json"
    write_json(test_data, tmp_path, overwrite=True)
    loaded = read_json(tmp_path)
    assert loaded == test_data, "JSON round-trip failed!"
    # Clean up test file
    (get_project_root() / tmp_path).unlink()
    print(f"[6] JSON round-trip OK")

    print("\n✅  All utils.py tests passed.")
