"""Evaluator for the bar-exams suite.

Scores multiple-choice law questions. Full credit requires a valid JSON
response ({"selected_option": ..., "explanation": ...}) with the correct
option; a correct answer given without the JSON format earns partial credit.
"""

import json
import re


def _extract_json(text: str):
    """Parse JSON from a response, tolerating code fences and surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_option(value: str) -> str:
    return value.strip().upper().rstrip(").")


def evaluate(prompt: str, response: str, expected: str | None = None, **kwargs) -> dict:
    """Scores an LLM response against the input prompt and optional expected output."""
    if expected is None:
        return {
            "passed": False,
            "score": 0.0,
            "reason": "bar-exams requires an expected option letter",
            "metadata": {},
        }

    # 1. Try the JSON format first (full credit path).
    try:
        data = _extract_json(response)
    except (json.JSONDecodeError, ValueError):
        data = None

    if isinstance(data, dict) and "selected_option" in data:
        selected = str(data["selected_option"])
        if _normalize_option(selected) == _normalize_option(expected):
            return {
                "passed": True,
                "score": 1.0,
                "reason": f"Correct option '{selected}' in valid JSON",
                "metadata": {"selected_option": selected, "explanation": data.get("explanation")},
            }
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Valid JSON but selected '{selected}' instead of '{expected}'",
            "metadata": {
                "selected_option": selected,
                "expected": expected,
                "explanation": data.get("explanation"),
            },
        }

    # 2. No valid JSON: look for the option letter in the raw text (partial credit).
    letter = _normalize_option(expected)
    if re.search(rf"\b{re.escape(letter)}\b", response):
        return {
            "passed": False,
            "score": 0.5,
            "reason": f"Answer '{letter}' present but response is not valid JSON",
            "metadata": {"response_preview": response[:200]},
        }
    return {
        "passed": False,
        "score": 0.0,
        "reason": "No valid JSON and the expected option was not found in the response",
        "metadata": {"response_preview": response[:200]},
    }
