"""
investigation/guardrails/actions.py
=====================================
Custom NeMo Guardrails Action functions.

Each function here maps to an `execute <action_name>` call in the
Colang flows defined in config.yml. They are registered with the
RailsConfig at runtime via register_action().

Guardrail actions are synchronous Python functions that receive
keyword arguments and return a primitive (bool, str, list).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Approved AML statute whitelist (mirrors the prompt instructions)
# ---------------------------------------------------------------------------
APPROVED_STATUTES = re.compile(
    r"31\s+CFR\s+§?1020|BSA|31\s+USC\s+§?5318|"
    r"FATF\s+Rec(?:ommendation)?s?\s*\d+|"
    r"L561-\d+|AMLD[56]|Directive\s+201[58]/\d+|"
    r"Wolfsberg|Basel\s+AML|CERFA\s+10534",
    re.IGNORECASE,
)

STATUTE_LIKE = re.compile(
    r"\b(?:§|Art\.|Article|Section|Sec\.|Regulation|Directive|CFR|USC|CMF)\s*[\d\-\.]+",
    re.IGNORECASE,
)

# PII patterns
PII_PATTERNS = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"        # SSN
    r"|\b[A-Z]{1,3}\d{6,9}\b"       # Passport (simplified)
    r"|\b\d{13,19}\b",               # Card / account numbers (raw)
    re.IGNORECASE,
)

# Prompt injection markers
INJECTION_MARKERS = [
    "ignore previous instructions",
    "disregard the above",
    "forget your system prompt",
    "act as",
    "jailbreak",
    "you are now",
]


# ---------------------------------------------------------------------------
# Action: PII detection
# ---------------------------------------------------------------------------

async def check_pii_action(text: str, **kwargs) -> bool:
    """
    Returns True if the prompt contains raw PII patterns that should
    not be sent to the LLM API.

    Note: in production, replace this regex check with a dedicated PII
    detection model (e.g. Microsoft Presidio or AWS Comprehend Medical).
    """
    return bool(PII_PATTERNS.search(text))


# ---------------------------------------------------------------------------
# Action: Prompt injection detection
# ---------------------------------------------------------------------------

async def detect_prompt_injection(text: str, **kwargs) -> bool:
    """
    Returns True if the input contains common prompt injection patterns.
    """
    text_lower = text.lower()
    return any(marker in text_lower for marker in INJECTION_MARKERS)


# ---------------------------------------------------------------------------
# Action: Statute whitelist validation
# ---------------------------------------------------------------------------

async def check_statute_whitelist(text: str, **kwargs) -> List[str]:
    """
    Find all statute-like references in `text` and return those that
    are NOT in the approved whitelist.

    Returns an empty list (falsy) if all citations are approved.
    Returns a list of violating strings (truthy) if any are not approved.
    """
    violations = []
    for match in STATUTE_LIKE.finditer(text):
        citation = match.group(0)
        if not APPROVED_STATUTES.search(citation):
            violations.append(citation)
    return violations


# ---------------------------------------------------------------------------
# Action: JSON schema validation
# ---------------------------------------------------------------------------

async def validate_json_schema(text: str, **kwargs) -> bool:
    """
    Returns True if the LLM output contains a parseable JSON object.
    The Colang flow checks `not $is_valid_json` to trigger reformatting.
    """
    import json, re
    # Try to extract JSON block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = (fence.group(1) if fence else None) or (brace.group(0) if brace else None)
    if candidate is None:
        return False
    try:
        json.loads(candidate)
        return True
    except json.JSONDecodeError:
        return False


# ---------------------------------------------------------------------------
# Action: Fabricated amount detection
# ---------------------------------------------------------------------------

async def detect_fabricated_amounts(
    response: str,
    context: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    Detect monetary amounts in the LLM response that do NOT appear in
    the evidence context string.

    Strategy: extract all dollar/euro amounts from response, then check
    if each is present (as a substring) in the context. Any amount not
    found in context is considered potentially fabricated.

    Limitation: this is a heuristic — amounts formatted differently
    ($9,500 vs $9500 vs 9,500 USD) may cause false positives.
    In production, normalise amounts before comparison.
    """
    if not context:
        return False   # Can't validate without context — skip

    AMOUNT_PATTERN = re.compile(
        r"\$[\d,]+(?:\.\d{2})?|\b[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|MAD)\b",
        re.IGNORECASE,
    )
    response_amounts = AMOUNT_PATTERN.findall(response)
    for amount in response_amounts:
        # Normalise: remove $ , and trailing .00 for comparison
        normalised = re.sub(r"[,$\s]", "", amount).rstrip("0").rstrip(".")
        if normalised not in context:
            return True   # Fabrication detected

    return False


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_all_actions(rails_config: Any) -> None:
    """
    Register all custom actions with a NeMo Guardrails RailsConfig.

    Call this after loading the RailsConfig:
        cfg = RailsConfig.from_path("investigation/guardrails/")
        register_all_actions(cfg)
        rails = LLMRails(cfg)
    """
    rails_config.register_action(check_pii_action,          name="check_pii_action")
    rails_config.register_action(detect_prompt_injection,   name="detect_prompt_injection")
    rails_config.register_action(check_statute_whitelist,   name="check_statute_whitelist")
    rails_config.register_action(validate_json_schema,      name="validate_json_schema")
    rails_config.register_action(detect_fabricated_amounts, name="detect_fabricated_amounts")
