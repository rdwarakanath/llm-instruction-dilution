"""
tokenizer_utils.py
------------------
Token counting and prompt budget enforcement.

Why tiktoken instead of the model's own tokenizer?
  - Ollama does not expose a tokenization API.
  - tiktoken's cl100k_base encoding (used by GPT-4) is a close approximation
    for LLaMA-family models and is deterministic and fast.
  - For this research we care about *relative* token counts across conditions,
    not exact counts. tiktoken is consistent and reproducible.

Responsibilities:
  - Count tokens in any string
  - Check if a prompt fits within a model's context window
  - Enforce per-section token budgets (system / user / total)
  - Return truncated text if a section exceeds its budget

Rules:
  - No circular imports — only imports from utils.py and stdlib/third-party
  - Encoder is loaded once at module level (expensive operation)
"""

import tiktoken

from utils import load_yaml


# ─────────────────────────────────────────────────────────────
# ENCODER  (loaded once at import time)
# ─────────────────────────────────────────────────────────────

# cl100k_base is the encoding used by GPT-4 / GPT-3.5.
# It is a good proxy for LLaMA 3.x tokenization for our purposes.
_ENCODER = tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────────────────────
# CORE COUNTING
# ─────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Returns the number of tokens in a string using cl100k_base encoding.

    Args:
        text: Any string — prompt section, full prompt, or raw text.

    Returns:
        Integer token count.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"[tokenizer_utils] count_tokens expects a str, got {type(text).__name__}"
        )
    return len(_ENCODER.encode(text))


def count_tokens_dict(sections: dict) -> dict:
    """
    Counts tokens for each section in a dict of {section_name: text}.

    Useful for logging per-section token usage.

    Args:
        sections: e.g. {'system': '...', 'user': '...', 'noise': '...'}

    Returns:
        Dict of {section_name: token_count}
    """
    if not isinstance(sections, dict):
        raise TypeError(
            f"[tokenizer_utils] count_tokens_dict expects a dict, got {type(sections).__name__}"
        )
    return {k: count_tokens(v) for k, v in sections.items()}


# ─────────────────────────────────────────────────────────────
# BUDGET ENFORCEMENT
# ─────────────────────────────────────────────────────────────

def fits_within_budget(text: str, max_tokens: int) -> bool:
    """
    Returns True if text fits within max_tokens, False otherwise.

    Args:
        text:       String to check.
        max_tokens: Maximum allowed token count.

    Returns:
        bool
    """
    return count_tokens(text) <= max_tokens


def enforce_budget(text: str, max_tokens: int, label: str = "section") -> str:
    """
    Truncates text to fit within max_tokens if it exceeds the budget.

    Truncation is token-exact: decodes the allowed tokens back to a string.
    A truncation warning marker is appended so it is detectable in outputs.

    Args:
        text:       Input string.
        max_tokens: Maximum allowed tokens.
        label:      Name of the section (for warning message).

    Returns:
        Original text if within budget, truncated text + marker if over.
    """
    tokens = _ENCODER.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated = _ENCODER.decode(tokens[:max_tokens])
    print(
        f"[tokenizer_utils] WARNING: '{label}' truncated from "
        f"{len(tokens)} to {max_tokens} tokens."
    )
    return truncated + " [TRUNCATED]"


# ─────────────────────────────────────────────────────────────
# PROMPT-LEVEL VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_prompt_budget(
    system_text: str,
    user_text: str,
    config_path: str = "config/prompt_config.yaml",
) -> dict:
    """
    Validates that system and user prompt sections are within configured limits.

    Loads token limits from prompt_config.yaml:
      - token_limits.system_prompt_max
      - token_limits.user_prompt_max
      - token_limits.total_max

    Args:
        system_text:  The system prompt string.
        user_text:    The user prompt string.
        config_path:  Path to prompt_config.yaml (relative to project root).

    Returns:
        A dict with:
          {
            'system_tokens':  int,
            'user_tokens':    int,
            'total_tokens':   int,
            'system_ok':      bool,
            'user_ok':        bool,
            'total_ok':       bool,
            'all_ok':         bool,
            'limits': {
                'system_max': int,
                'user_max':   int,
                'total_max':  int,
            }
          }

    Raises:
        KeyError: If token_limits section is missing from prompt_config.yaml.
    """
    cfg = load_yaml(config_path)

    try:
        limits = cfg["token_limits"]
        sys_max   = limits["system_prompt_max"]
        user_max  = limits["user_prompt_max"]
        total_max = limits["total_max"]
    except KeyError as e:
        raise KeyError(
            f"[tokenizer_utils] Missing key in prompt_config.yaml under "
            f"'token_limits': {e}"
        )

    sys_tokens  = count_tokens(system_text)
    user_tokens = count_tokens(user_text)
    total       = sys_tokens + user_tokens

    sys_ok   = sys_tokens  <= sys_max
    user_ok  = user_tokens <= user_max
    total_ok = total       <= total_max

    return {
        "system_tokens": sys_tokens,
        "user_tokens":   user_tokens,
        "total_tokens":  total,
        "system_ok":     sys_ok,
        "user_ok":       user_ok,
        "total_ok":      total_ok,
        "all_ok":        sys_ok and user_ok and total_ok,
        "limits": {
            "system_max": sys_max,
            "user_max":   user_max,
            "total_max":  total_max,
        },
    }


def check_context_window(
    prompt_tokens: int,
    model_id: str,
    config_path: str = "config/models.yaml",
) -> dict:
    """
    Checks if a prompt fits within a model's context window.

    Args:
        prompt_tokens: Total token count of the prompt.
        model_id:      Model id string, e.g. 'model_3b' or 'model_8b'.
        config_path:   Path to models.yaml (relative to project root).

    Returns:
        {
            'fits':           bool,
            'prompt_tokens':  int,
            'context_window': int,
            'headroom':       int   (context_window - prompt_tokens),
            'model_id':       str,
        }

    Raises:
        ValueError: If model_id not found in models.yaml.
    """
    cfg = load_yaml(config_path)
    models = cfg.get("models", [])

    model_cfg = next((m for m in models if m["id"] == model_id), None)
    if model_cfg is None:
        valid_ids = [m["id"] for m in models]
        raise ValueError(
            f"[tokenizer_utils] Model ID '{model_id}' not found in models.yaml.\n"
            f"  Valid IDs: {valid_ids}"
        )

    context_window = model_cfg["context_window"]
    fits           = prompt_tokens <= context_window
    headroom       = context_window - prompt_tokens

    return {
        "fits":           fits,
        "prompt_tokens":  prompt_tokens,
        "context_window": context_window,
        "headroom":       headroom,
        "model_id":       model_id,
    }


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: FULL PROMPT TOKEN SUMMARY
# ─────────────────────────────────────────────────────────────

def prompt_token_summary(
    system_text: str,
    user_text: str,
    model_id: str,
) -> dict:
    """
    Returns a complete token usage summary for a prompt.
    Combines validate_prompt_budget() and check_context_window().

    Args:
        system_text: System prompt string.
        user_text:   User prompt string.
        model_id:    e.g. 'model_3b'

    Returns:
        Combined dict with all budget and context window checks.
    """
    budget = validate_prompt_budget(system_text, user_text)
    ctx    = check_context_window(budget["total_tokens"], model_id)

    return {
        **budget,
        "context_window_check": ctx,
    }


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run directly: python src/tokenizer_utils.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== tokenizer_utils.py self-test ===\n")

    # 1. Basic counting
    text = "Ravi Kumar is a 21-year-old engineering student from Chennai."
    n = count_tokens(text)
    print(f"[1] Token count for sample sentence: {n} tokens")
    assert n > 0

    # 2. count_tokens_dict
    sections = {
        "skill":  "You are a precise JSON extractor.",
        "noise":  "Some irrelevant background text goes here.",
        "input":  text,
    }
    counts = count_tokens_dict(sections)
    print(f"[2] Section counts: {counts}")
    assert all(v > 0 for v in counts.values())

    # 3. fits_within_budget
    assert fits_within_budget(text, 100) is True
    assert fits_within_budget(text, 1) is False
    print(f"[3] fits_within_budget OK")

    # 4. enforce_budget — force truncation
    long_text = "word " * 200  # ~200 tokens
    truncated = enforce_budget(long_text, max_tokens=50, label="test_section")
    assert "[TRUNCATED]" in truncated
    assert count_tokens(truncated) <= 55  # small tolerance for marker
    print(f"[4] enforce_budget truncation OK")

    # 5. validate_prompt_budget
    result = validate_prompt_budget(
        system_text="You extract JSON from text.",
        user_text=text,
    )
    print(f"[5] validate_prompt_budget: {result}")
    assert "all_ok" in result
    assert result["system_tokens"] > 0
    assert result["user_tokens"] > 0

    # 6. check_context_window
    ctx = check_context_window(prompt_tokens=500, model_id="model_3b")
    print(f"[6] context_window check (model_3b, 500 tokens): {ctx}")
    assert ctx["fits"] is True
    assert ctx["context_window"] == 4096

    # 7. prompt_token_summary
    summary = prompt_token_summary(
        system_text="You extract JSON from text.",
        user_text=text,
        model_id="model_8b",
    )
    print(f"[7] prompt_token_summary all_ok: {summary['all_ok']}")

    # 8. Type error handling
    try:
        count_tokens(12345)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"[8] TypeError correctly raised: {e}")

    print("\n✅  All tokenizer_utils.py tests passed.")
