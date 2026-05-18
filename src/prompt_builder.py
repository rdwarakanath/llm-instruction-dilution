"""
prompt_builder.py
-----------------
Builds prompt strings from templates, skill blocks, and noise blocks.

This is the CORE of the research experiment.
The three prompt strategies are:

  1. monolithic         — all instructions in one block, noise included
  2. modular            — skill and output format separated from noise
  3. system_separated   — skill+format in system role, input in user role only

For Ollama's API, prompts are returned as:
  {
      'system': '...',   # passed as system role (can be empty string)
      'user':   '...',   # passed as user role
  }

Responsibilities:
  - Load template files from prompts/
  - Load skill blocks from skills/
  - Load noise blocks from skills/noise_blocks/
  - Render templates by replacing placeholders
  - Validate all placeholders are filled before returning
  - Return token counts alongside the built prompt

Rules:
  - Raises explicit errors if any placeholder is unfilled
  - Imports only from utils.py, tokenizer_utils.py, and stdlib
"""

import re
from typing import Optional

from tokenizer_utils import count_tokens, count_tokens_dict, validate_prompt_budget
from utils import load_yaml, read_text


# ─────────────────────────────────────────────────────────────
# PLACEHOLDER NAMES  (must match what's in template files)
# ─────────────────────────────────────────────────────────────

SKILL_BLOCK   = "{{SKILL_BLOCK}}"
NOISE_BLOCK   = "{{NOISE_BLOCK}}"
USER_INPUT    = "{{USER_INPUT}}"
OUTPUT_FORMAT = "{{OUTPUT_FORMAT}}"

ALL_PLACEHOLDERS = [SKILL_BLOCK, NOISE_BLOCK, USER_INPUT, OUTPUT_FORMAT]

# Output format strings injected per task type
OUTPUT_FORMATS = {
    "json_task": (
        "Return ONLY valid JSON. No explanation. No markdown. "
        "No extra text before or after the JSON object."
    ),
    "struct_task": (
        "Return ONLY the formatted output (email or report). "
        "No explanation. No markdown fences. No extra commentary."
    ),
}

# Noise level label → file path
NOISE_FILE_MAP = {
    "short":  "skills/noise_blocks/noise_300.txt",
    "medium": "skills/noise_blocks/noise_800.txt",
    "long":   "skills/noise_blocks/noise_1500.txt",
}

# Skill file paths per task type
SKILL_FILE_MAP = {
    "json_task":   "skills/json_skill.txt",
    "struct_task": "skills/email_skill.txt",
}

# Template file paths per strategy
TEMPLATE_FILE_MAP = {
    "monolithic":        "prompts/monolithic/template_v1.txt",
    "modular":           "prompts/modular/template_v1.txt",
    "system_separated":  "prompts/system_separated/template_v1.txt",
}

VALID_STRATEGIES   = set(TEMPLATE_FILE_MAP.keys())
VALID_NOISE_LEVELS = set(NOISE_FILE_MAP.keys())
VALID_TASK_TYPES   = set(SKILL_FILE_MAP.keys())


# ─────────────────────────────────────────────────────────────
# TEMPLATE RENDERING
# ─────────────────────────────────────────────────────────────

def _render_template(template: str, replacements: dict) -> str:
    """
    Replaces all placeholder keys in template with their values.

    Args:
        template:     Raw template string with {{PLACEHOLDER}} markers.
        replacements: Dict mapping placeholder string to replacement value,
                      e.g. {'{{SKILL_BLOCK}}': '...', '{{USER_INPUT}}': '...'}

    Returns:
        Rendered string.

    Raises:
        ValueError: If any placeholder in ALL_PLACEHOLDERS is still present
                    after rendering.
    """
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    # Check for any unfilled placeholders
    unfilled = [p for p in ALL_PLACEHOLDERS if p in result]
    if unfilled:
        raise ValueError(
            f"[prompt_builder] Unfilled placeholders remain after rendering: {unfilled}\n"
            f"  Provided replacements: {list(replacements.keys())}"
        )

    return result.strip()


def _find_section(text: str, marker: str) -> str:
    """
    Extracts the content under a [SECTION] marker from a system_separated template.

    Args:
        text:   Full rendered template string.
        marker: e.g. '[SYSTEM]' or '[USER]'

    Returns:
        Content under that marker, stripped.

    Raises:
        ValueError: If the marker is not found.
    """
    pattern = rf"\[{re.escape(marker.strip('[]'))}\](.*?)(?=\[[A-Z]+\]|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(
            f"[prompt_builder] Marker '{marker}' not found in system_separated template.\n"
            f"  Template content (first 200 chars): {text[:200]}"
        )
    return match.group(1).strip()


# ─────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────

def build_prompt(
    strategy: str,
    task_type: str,
    user_input: str,
    noise_level: str,
    model_id: str = "model_3b",
    validate_tokens: bool = True,
) -> dict:
    """
    Builds a complete prompt for a given strategy, task type, and noise level.

    Args:
        strategy:         'monolithic', 'modular', or 'system_separated'
        task_type:        'json_task' or 'struct_task'
        user_input:       The input sentence/brief from the dataset record.
        noise_level:      'short', 'medium', or 'long'
        model_id:         Model ID for context window check (e.g. 'model_3b')
        validate_tokens:  If True, logs token budget status (does NOT hard-fail).

    Returns:
        {
            'system':       str,   # system role content (empty for monolithic/modular)
            'user':         str,   # user role content
            'strategy':     str,
            'task_type':    str,
            'noise_level':  str,
            'token_counts': {
                'system':  int,
                'user':    int,
                'total':   int,
            },
            'budget_ok':    bool,
        }

    Raises:
        ValueError: For unknown strategy, task_type, or noise_level.
        FileNotFoundError: If any required file is missing.
    """
    # ── input validation ──
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"[prompt_builder] Unknown strategy: '{strategy}'. "
            f"Must be one of: {sorted(VALID_STRATEGIES)}"
        )
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"[prompt_builder] Unknown task_type: '{task_type}'. "
            f"Must be one of: {sorted(VALID_TASK_TYPES)}"
        )
    if noise_level not in VALID_NOISE_LEVELS:
        raise ValueError(
            f"[prompt_builder] Unknown noise_level: '{noise_level}'. "
            f"Must be one of: {sorted(VALID_NOISE_LEVELS)}"
        )
    if not user_input or not user_input.strip():
        raise ValueError("[prompt_builder] user_input must be a non-empty string.")

    # ── load building blocks ──
    template   = read_text(TEMPLATE_FILE_MAP[strategy])
    skill      = read_text(SKILL_FILE_MAP[task_type])
    noise      = read_text(NOISE_FILE_MAP[noise_level])
    out_format = OUTPUT_FORMATS[task_type]

    # ── build replacements dict ──
    replacements = {
        SKILL_BLOCK:   skill,
        NOISE_BLOCK:   noise,
        USER_INPUT:    user_input.strip(),
        OUTPUT_FORMAT: out_format,
    }

    # ── render based on strategy ──
    if strategy in ("monolithic", "modular"):
        # These strategies use a single-block template
        # system role is empty; everything goes in user role
        rendered = _render_template(template, replacements)
        system_text = ""
        user_text   = rendered

    elif strategy == "system_separated":
        # Template has [SYSTEM] and [USER] markers
        # For system_separated: noise block is NOT injected (per research design)
        # Replace NOISE_BLOCK with empty string
        replacements[NOISE_BLOCK] = ""
        rendered    = _render_template(template, replacements)
        system_text = _find_section(rendered, "[SYSTEM]")
        user_text   = _find_section(rendered, "[USER]")

    # ── token counts ──
    token_counts = {
        "system": count_tokens(system_text),
        "user":   count_tokens(user_text),
        "total":  count_tokens(system_text) + count_tokens(user_text),
    }

    # ── budget check ──
    budget_ok = True
    if validate_tokens:
        budget_result = validate_prompt_budget(system_text, user_text)
        budget_ok = budget_result["all_ok"]
        if not budget_ok:
            print(
                f"[prompt_builder] WARNING: Token budget exceeded for "
                f"strategy='{strategy}', task='{task_type}', noise='{noise_level}'.\n"
                f"  Counts: {token_counts}\n"
                f"  Limits: {budget_result['limits']}"
            )

    return {
        "system":      system_text,
        "user":        user_text,
        "strategy":    strategy,
        "task_type":   task_type,
        "noise_level": noise_level,
        "token_counts": token_counts,
        "budget_ok":   budget_ok,
    }


# ─────────────────────────────────────────────────────────────
# BATCH BUILD
# ─────────────────────────────────────────────────────────────

def build_prompts_for_record(
    record: dict,
    task_type: str,
    strategies: Optional[list] = None,
    noise_levels: Optional[list] = None,
    model_id: str = "model_3b",
) -> list[dict]:
    """
    Builds all prompt variants for a single dataset record.

    Args:
        record:       A single record dict from data_loader.load_dataset().
        task_type:    'json_task' or 'struct_task'
        strategies:   List of strategies to use. Default: all three.
        noise_levels: List of noise levels to use. Default: all three.
        model_id:     For token budget checking.

    Returns:
        List of prompt dicts, one per (strategy × noise_level) combination.
        Each dict also includes the original record id.
    """
    if strategies is None:
        strategies = sorted(VALID_STRATEGIES)
    if noise_levels is None:
        noise_levels = sorted(VALID_NOISE_LEVELS)

    user_input = record["input"]
    record_id  = record["id"]

    results = []
    for strategy in strategies:
        for noise_level in noise_levels:
            prompt = build_prompt(
                strategy=strategy,
                task_type=task_type,
                user_input=user_input,
                noise_level=noise_level,
                model_id=model_id,
            )
            prompt["record_id"] = record_id
            results.append(prompt)

    return results


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run directly: python src/prompt_builder.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== prompt_builder.py self-test ===\n")

    SAMPLE_INPUT = "Ravi Kumar is a 21-year-old engineering student from Chennai studying at Anna University."

    # 1. Build all three strategies × all three noise levels
    for strategy in ["monolithic", "modular", "system_separated"]:
        for noise in ["short", "medium", "long"]:
            result = build_prompt(
                strategy=strategy,
                task_type="json_task",
                user_input=SAMPLE_INPUT,
                noise_level=noise,
            )
            print(
                f"  [{strategy:20s} | {noise:6s}] "
                f"sys={result['token_counts']['system']:4d}t  "
                f"usr={result['token_counts']['user']:4d}t  "
                f"total={result['token_counts']['total']:4d}t  "
                f"budget_ok={result['budget_ok']}"
            )
            assert result["system"] is not None
            assert result["user"] is not None
            assert result["token_counts"]["total"] > 0

    print()

    # 2. system_separated should have non-empty system and user
    sep = build_prompt("system_separated", "json_task", SAMPLE_INPUT, "short")
    assert sep["system"] != "", "system_separated: system should not be empty"
    assert sep["user"]   != "", "system_separated: user should not be empty"
    print(f"[2] system_separated split OK: sys={sep['token_counts']['system']}t  usr={sep['token_counts']['user']}t ✓")

    # 3. monolithic should have empty system
    mono = build_prompt("monolithic", "json_task", SAMPLE_INPUT, "short")
    assert mono["system"] == "", "monolithic: system should be empty"
    print(f"[3] monolithic system empty OK ✓")

    # 4. struct_task works
    struct = build_prompt("modular", "struct_task", "Write a formal email declining a meeting.", "medium")
    assert struct["token_counts"]["total"] > 0
    print(f"[4] struct_task prompt built OK: {struct['token_counts']['total']} total tokens ✓")

    # 5. Invalid strategy raises
    try:
        build_prompt("bad_strategy", "json_task", SAMPLE_INPUT, "short")
        assert False, "Should raise"
    except ValueError:
        print(f"[5] Invalid strategy raises ValueError ✓")

    # 6. Invalid noise level raises
    try:
        build_prompt("monolithic", "json_task", SAMPLE_INPUT, "extreme")
        assert False, "Should raise"
    except ValueError:
        print(f"[6] Invalid noise_level raises ValueError ✓")

    # 7. batch build for a record
    record = {
        "id": "json_001",
        "input": SAMPLE_INPUT,
        "expected_output": {"name": "Ravi Kumar"},
        "metadata": {"fields_present": ["name"], "difficulty": "easy", "subtype": "test"},
    }
    prompts = build_prompts_for_record(record, "json_task")
    assert len(prompts) == 9, f"Expected 9 (3×3) prompts, got {len(prompts)}"
    print(f"[7] build_prompts_for_record: {len(prompts)} prompts generated (3 strategies × 3 noise levels) ✓")

    print("\n✅  All prompt_builder.py tests passed.")
