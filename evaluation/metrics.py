"""
metrics.py
----------
Core metric functions used by json_eval.py, structured_eval.py,
and stability_eval.py.

All functions are pure — no file I/O, no side effects.
Every function returns a float in [0.0, 1.0] unless documented otherwise.

Metrics implemented:
  - exact_match          : 1.0 if outputs identical, else 0.0
  - field_precision      : fraction of predicted fields that are correct
  - field_recall         : fraction of expected fields that were found
  - field_f1             : harmonic mean of precision and recall
  - format_compliance    : checks required structural sections are present
  - json_validity        : checks if a string parses as valid JSON
  - rouge_l              : longest common subsequence overlap
  - response_variance    : variance across repeated runs (stability)
  - normalise_text       : lowercase + strip for fair comparison
"""

import json
import math
import re
import string
from typing import Any


# ─────────────────────────────────────────────────────────────
# TEXT NORMALISATION
# ─────────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """
    Lowercase, strip whitespace, remove punctuation except @ . _
    Used before string comparisons to avoid false negatives
    from capitalisation or spacing differences.

    Args:
        text: Raw string.

    Returns:
        Normalised string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Keep @ . _ - for emails and names; remove other punctuation
    keep = set(string.ascii_lowercase + string.digits + " @._-")
    return "".join(c for c in text if c in keep)


# ─────────────────────────────────────────────────────────────
# EXACT MATCH
# ─────────────────────────────────────────────────────────────

def exact_match(predicted: str, expected: str) -> float:
    """
    Returns 1.0 if predicted == expected after normalisation, else 0.0.

    Args:
        predicted: Model output string.
        expected:  Ground truth string.

    Returns:
        1.0 or 0.0
    """
    return 1.0 if normalise_text(predicted) == normalise_text(expected) else 0.0


# ─────────────────────────────────────────────────────────────
# JSON VALIDITY
# ─────────────────────────────────────────────────────────────

def json_validity(response_text: str) -> dict:
    """
    Checks whether a model response is valid JSON.

    Handles common model misbehaviours:
      - Markdown code fences (```json ... ```)
      - Leading/trailing whitespace or newlines
      - JSON embedded inside prose

    Args:
        response_text: Raw model output string.

    Returns:
        {
            'valid':   bool,
            'parsed':  dict | None,   # parsed JSON if valid
            'clean':   str,           # cleaned string that was attempted
            'reason':  str,           # failure reason if not valid
        }
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return {
            "valid": False, "parsed": None,
            "clean": "", "reason": "Empty or non-string response."
        }

    # Step 1: strip markdown fences
    clean = response_text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    clean = clean.strip()

    # Step 2: try direct parse
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return {"valid": True, "parsed": parsed, "clean": clean, "reason": ""}
        else:
            return {
                "valid": False, "parsed": None, "clean": clean,
                "reason": f"Parsed JSON is not a dict (got {type(parsed).__name__})."
            }
    except json.JSONDecodeError:
        pass

    # Step 3: try to extract JSON object from within prose
    match = re.search(r"\{[^{}]*\}", clean, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return {
                    "valid": True, "parsed": parsed,
                    "clean": candidate,
                    "reason": "Extracted from prose."
                }
        except json.JSONDecodeError:
            pass

    return {
        "valid": False, "parsed": None, "clean": clean,
        "reason": "Response could not be parsed as a JSON object."
    }


# ─────────────────────────────────────────────────────────────
# FIELD-LEVEL METRICS  (for JSON extraction task)
# ─────────────────────────────────────────────────────────────

def _normalise_value(val: Any) -> str:
    """Converts a field value to a normalised string for comparison."""
    if val is None:
        return ""
    return normalise_text(str(val))


def field_precision(predicted: dict, expected: dict) -> float:
    """
    Fraction of fields in predicted that are also correct in expected.

    A field is correct if:
      - The key exists in expected, AND
      - The normalised values match.

    Args:
        predicted: Parsed JSON dict from model output.
        expected:  Ground truth dict.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if predicted is empty.
    """
    if not predicted:
        return 0.0

    correct = 0
    for key, val in predicted.items():
        if key in expected:
            if _normalise_value(val) == _normalise_value(expected[key]):
                correct += 1

    return correct / len(predicted)


def field_recall(predicted: dict, expected: dict) -> float:
    """
    Fraction of fields in expected that were correctly found in predicted.

    Args:
        predicted: Parsed JSON dict from model output.
        expected:  Ground truth dict.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if expected is empty.
    """
    if not expected:
        return 0.0

    correct = 0
    for key, val in expected.items():
        if key in predicted:
            if _normalise_value(predicted[key]) == _normalise_value(val):
                correct += 1

    return correct / len(expected)


def field_f1(predicted: dict, expected: dict) -> float:
    """
    Harmonic mean of field_precision and field_recall.

    Args:
        predicted: Parsed JSON dict from model output.
        expected:  Ground truth dict.

    Returns:
        Float in [0.0, 1.0].
    """
    p = field_precision(predicted, expected)
    r = field_recall(predicted, expected)

    if p + r == 0.0:
        return 0.0

    return 2 * p * r / (p + r)


def hallucination_rate(predicted: dict, expected: dict) -> float:
    """
    Fraction of predicted fields that are NOT in the expected output.
    A hallucinated field is one the model invented that does not exist
    in the ground truth.

    Args:
        predicted: Parsed JSON dict from model output.
        expected:  Ground truth dict.

    Returns:
        Float in [0.0, 1.0]. 0.0 means no hallucinations.
    """
    if not predicted:
        return 0.0

    hallucinated = sum(1 for k in predicted if k not in expected)
    return hallucinated / len(predicted)


# ─────────────────────────────────────────────────────────────
# FORMAT COMPLIANCE  (for structured text task)
# ─────────────────────────────────────────────────────────────

def format_compliance(response_text: str, required_sections: list) -> dict:
    """
    Checks whether required structural sections are present in the response.

    For emails: checks for subject, greeting, body, sign_off.
    For reports: checks for title and at least one content section.

    Matching is case-insensitive and uses keyword presence, not exact match.

    Args:
        response_text:     Raw model output string.
        required_sections: List of section names from dataset metadata,
                           e.g. ['subject', 'greeting', 'body', 'sign_off']

    Returns:
        {
            'score':            float,   # fraction of sections present
            'sections_found':   list,
            'sections_missing': list,
            'has_markdown_fence': bool,  # True if model used ``` (penalised)
        }
    """
    if not response_text or not response_text.strip():
        return {
            "score": 0.0,
            "sections_found": [],
            "sections_missing": required_sections,
            "has_markdown_fence": False,
        }

    text_lower = response_text.lower()

    # Check for markdown fences (model was told not to use them)
    has_fence = "```" in response_text

    # Section detection keywords
    SECTION_KEYWORDS = {
        "subject":        ["subject:"],
        "greeting":       ["dear ", "hello ", "hi "],
        "body":           None,          # inferred from length
        "sign_off":       ["regards,", "sincerely,", "yours", "warm regards",
                           "best regards", "with regards", "respectfully",
                           "thank you,", "with gratitude", "with warmth",
                           "with pride"],
        "title":          None,          # inferred from first non-empty line
        "summary":        ["summary", "overview", "background"],
        "cause":          ["cause", "reason", "root cause"],
        "resolution":     ["resolution", "resolved", "action taken", "fix"],
        "findings":       ["finding", "result", "observation"],
        "recommendation": ["recommend", "suggestion", "proposed"],
        "action_items":   ["action item", "next step", "follow-up"],
        "decisions":      ["decision", "agreed", "approved", "resolved"],
        "objective":      ["objective", "purpose", "goal", "scope"],
        "progress":       ["progress", "completed", "done", "achieved"],
        "blockers":       ["blocker", "issue", "challenge", "delay", "risk"],
        "next_steps":     ["next step", "next week", "planned", "upcoming"],
        "highlights":     ["highlight", "positive", "strength", "what went well"],
        "concerns":       ["concern", "issue", "challenge", "weakness"],
        "status":         ["status:", "status —", "overall:", "rating:"],
        "impact":         ["impact", "beneficiar", "outcome", "result"],
        "outline":        ["outline", "structure", "section"],
        "strengths":      ["strength", "positive", "what went well", "commend"],
        "improvement_areas": ["improve", "weakness", "area for", "gap"],
        "development_plan":  ["plan", "recommend", "next step", "workshop"],
        "risks":          ["risk", "threat", "vulnerability"],
        "conclusion":     ["conclusion", "overall", "in summary"],
        "breakdown":      ["breakdown", "distribution", "%", "percent"],
        "profile":        ["profile", "overview", "background"],
        "transferred":    ["transferred", "handed over", "completed"],
        "pending":        ["pending", "outstanding", "remaining"],
        "financial_status": ["budget", "financial", "cost", "expenditure"],
        "discharge_instructions": ["instruction", "follow-up", "advised", "prescription"],
        "diagnosis":      ["diagnosis", "condition", "finding"],
        "treatment":      ["treatment", "therapy", "medication", "prescribed"],
        "activities":     ["activit", "programme", "initiative", "camp"],
        "financials":     ["expenditure", "utilised", "amount", "disbursed"],
        "topics":         ["topic", "covered", "module", "session"],
        "outcome":        ["outcome", "result", "achieved", "impact"],
        "gaps":           ["gap", "missing", "insufficient", "weak"],
        "recommendations": ["recommend", "suggest", "propose"],
        "change_summary": ["change", "revised", "updated", "effective"],
        "acknowledgement":["acknowledge", "understand", "recogni"],
        "reason":         ["reason", "cause", "due to", "because"],
        "feedback":       ["feedback", "comment", "rating", "score"],
        "demand":         ["demand", "forecast", "expected", "project"],
        "capacity":       ["capacity", "supply", "available", "current"],
        "gap":            ["gap", "shortfall", "deficit", "insufficient"],
        "situation":      ["situation", "current", "context", "background"],
        "proposal":       ["propose", "recommend", "suggest"],
        "savings":        ["saving", "reduc", "cost", "benefit"],
        "evaluation":     ["evaluat", "assess", "compar", "criteria"],
        "analysis":       ["analysis", "finding", "result", "observation"],
        "pipeline_summary": ["stage", "application", "interview", "shortlist"],
        "observations":   ["observation", "note", "finding"],
        "actions":        ["action", "step", "taken", "done"],
        "key_driver":     ["driver", "factor", "due to", "attributed"],
        "quote":          ["said", "stated", "commented", "noted"],
        "outlook":        ["outlook", "forecast", "project", "next"],
        "violations":     ["violation", "issue", "non-compliant", "finding"],
        "follow_up":      ["follow-up", "re-inspection", "monitor", "next"],
        "completion":     ["complet", "finish", "done", "achiev"],
        "simulation":     ["simulat", "phishing", "test", "exercise"],
        "participation":  ["participat", "attend", "enrol", "complet"],
        "next_phase":     ["next phase", "phase 2", "q2", "upcoming"],
        "channels":       ["channel", "source", "platform", "media"],
        "nps":            ["nps", "net promoter", "score", "satisfaction"],
        "satisfaction":   ["satisf", "positive", "well", "strength"],
        "concern":        ["concern", "issue", "pain point", "challeng"],
        "what_went_well": ["went well", "positive", "strength", "success"],
        "what_went_wrong":["went wrong", "issue", "problem", "challeng"],
        "lessons":        ["lesson", "learn", "takeaway", "improve"],
        "publications":   ["publicat", "paper", "journal", "conference"],
        "projects":       ["project", "funded", "grant", "research"],
        "phd":            ["phd", "doctoral", "degree", "award"],
        "regional_performance": ["region", "south", "north", "zone"],
        "assessment":     ["assess", "rating", "evaluat", "overall"],
        "change_summary": ["change", "revised", "updated", "effective"],
        "plan":           ["plan", "phase", "step", "action"],
        "facility":       ["facilit", "office", "readiness", "prepared"],
        "sentiment":      ["sentiment", "survey", "feeling", "response"],
        "root_cause":     ["root cause", "cause", "reason", "why"],
        "action_plan":    ["action plan", "plan", "step", "measure"],
        "deadline":       ["deadline", "by ", "within", "date"],
    }

    found = []
    missing = []

    for section in required_sections:
        section_lower = section.lower()
        keywords = SECTION_KEYWORDS.get(section_lower)

        if keywords is None:
            # Infer presence by content length (body/title)
            if section_lower == "body":
                # Body present if response has at least 20 words total
                # (emails/reports are always longer than greeting alone)
                word_count = len(response_text.split())
                found.append(section) if word_count >= 20 else missing.append(section)
            elif section_lower == "title":
                # Title present if first non-empty line is short (< 15 words)
                # and does NOT start with 'subject:' (that is the subject section)
                lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
                if (lines
                        and len(lines[0].split()) < 15
                        and not lines[0].lower().startswith("subject:")):
                    found.append(section)
                else:
                    missing.append(section)
            else:
                # Unknown section — check if section name itself appears
                if section_lower in text_lower:
                    found.append(section)
                else:
                    missing.append(section)
        else:
            if any(kw in text_lower for kw in keywords):
                found.append(section)
            else:
                missing.append(section)

    score = len(found) / len(required_sections) if required_sections else 0.0

    return {
        "score":             score,
        "sections_found":    found,
        "sections_missing":  missing,
        "has_markdown_fence": has_fence,
    }


# ─────────────────────────────────────────────────────────────
# ROUGE-L  (for structured text task)
# ─────────────────────────────────────────────────────────────

def _lcs_length(a: list, b: list) -> int:
    """Computes the length of the Longest Common Subsequence of two lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0

    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    return prev[n]


def rouge_l(predicted: str, reference: str) -> dict:
    """
    Computes ROUGE-L score (based on Longest Common Subsequence of words).

    Args:
        predicted:  Model output string.
        reference:  Reference output string.

    Returns:
        {
            'precision': float,
            'recall':    float,
            'f1':        float,
        }
    """
    pred_tokens = normalise_text(predicted).split()
    ref_tokens  = normalise_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)

    precision = lcs / len(pred_tokens)
    recall    = lcs / len(ref_tokens)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────
# STABILITY  (variance across repeated runs)
# ─────────────────────────────────────────────────────────────

def response_variance(scores: list) -> dict:
    """
    Computes variance and standard deviation of a list of scores
    across repeated runs of the same prompt cell.

    Used by stability_eval.py to measure output consistency.

    Args:
        scores: List of floats (e.g. f1 scores from 3 repeated runs).

    Returns:
        {
            'mean':   float,
            'std':    float,
            'var':    float,
            'min':    float,
            'max':    float,
            'range':  float,
            'n':      int,
        }

    Raises:
        ValueError: If scores list is empty.
    """
    if not scores:
        raise ValueError("[metrics] response_variance: scores list is empty.")

    n    = len(scores)
    mean = sum(scores) / n
    var  = sum((s - mean) ** 2 for s in scores) / n
    std  = math.sqrt(var)

    return {
        "mean":  round(mean, 4),
        "std":   round(std, 4),
        "var":   round(var, 4),
        "min":   round(min(scores), 4),
        "max":   round(max(scores), 4),
        "range": round(max(scores) - min(scores), 4),
        "n":     n,
    }


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python src/metrics.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== metrics.py self-test ===\n")

    # 1. normalise_text
    assert normalise_text("  Ravi Kumar! ") == "ravi kumar"
    assert normalise_text("ravi.kumar@zoho.com") == "ravi.kumar@zoho.com"
    print("[1] normalise_text OK ✓")

    # 2. exact_match
    assert exact_match("Ravi Kumar", "ravi kumar") == 1.0
    assert exact_match("Ravi", "Priya") == 0.0
    print("[2] exact_match OK ✓")

    # 3. json_validity — valid
    r = json_validity('{"name": "Ravi", "age": 21}')
    assert r["valid"] is True
    assert r["parsed"]["name"] == "Ravi"
    print("[3] json_validity (valid) OK ✓")

    # 4. json_validity — markdown fence
    r = json_validity('```json\n{"name": "Ravi"}\n```')
    assert r["valid"] is True
    print("[4] json_validity (markdown fence) OK ✓")

    # 5. json_validity — prose wrapper
    r = json_validity('Here is the JSON: {"name": "Ravi", "age": 21} as requested.')
    assert r["valid"] is True
    print("[5] json_validity (prose wrapper) OK ✓")

    # 6. json_validity — invalid
    r = json_validity("This is not JSON at all.")
    assert r["valid"] is False
    print("[6] json_validity (invalid) OK ✓")

    # 7. field_precision / recall / f1
    pred = {"name": "Ravi Kumar", "age": "21", "city": "Chennai"}
    exp  = {"name": "Ravi Kumar", "age": 21, "city": "Chennai", "email": "r@r.com"}
    p = field_precision(pred, exp)
    r_val = field_recall(pred, exp)
    f1 = field_f1(pred, exp)
    assert p == 1.0,              f"Expected precision 1.0, got {p}"
    assert round(r_val, 4) == 0.75, f"Expected recall 0.75, got {r_val}"
    assert f1 > 0.8
    print(f"[7] field metrics OK — P={p:.2f} R={r_val:.2f} F1={f1:.2f} ✓")

    # 8. hallucination_rate
    pred_hall = {"name": "Ravi", "age": 21, "hobby": "cricket"}
    exp_hall  = {"name": "Ravi", "age": 21}
    h = hallucination_rate(pred_hall, exp_hall)
    assert round(h, 4) == round(1/3, 4), f"Expected ~0.333, got {h}"
    print(f"[8] hallucination_rate OK — {h:.3f} ✓")

    # 9. format_compliance — email
    email_resp = """Subject: Leave Request

Dear Manager,

I am writing to request leave for five days due to a family emergency.

Thank you for your understanding.

Best regards,
Priya"""
    fc = format_compliance(email_resp, ["subject", "greeting", "body", "sign_off"])
    assert fc["score"] == 1.0, f"Expected 1.0, got {fc['score']}, missing={fc['sections_missing']}"
    print(f"[9] format_compliance (email) OK — score={fc['score']} ✓")

    # 10. format_compliance — markdown fence penalised
    fenced = "```\nSubject: Test\nDear Sir,\nBody.\nRegards,\nName\n```"
    fc2 = format_compliance(fenced, ["subject", "greeting", "body", "sign_off"])
    assert fc2["has_markdown_fence"] is True
    print(f"[10] format_compliance (markdown fence detected) OK ✓")

    # 11. rouge_l
    rl = rouge_l("the cat sat on the mat", "the cat sat on the mat")
    assert rl["f1"] == 1.0
    rl2 = rouge_l("the cat sat", "the dog sat on the mat")
    assert 0.0 < rl2["f1"] < 1.0
    print(f"[11] rouge_l OK — perfect={rl['f1']:.2f}, partial={rl2['f1']:.2f} ✓")

    # 12. response_variance
    stats = response_variance([0.9, 0.85, 0.92])
    assert stats["n"] == 3
    assert 0.88 < stats["mean"] < 0.90
    print(f"[12] response_variance OK — mean={stats['mean']}, std={stats['std']} ✓")

    # 13. empty scores raises
    try:
        response_variance([])
        assert False
    except ValueError:
        print("[13] response_variance empty list raises ValueError ✓")

    print("\n✅  All metrics.py tests passed.")