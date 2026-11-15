"""Evaluator for the json_format suite.

Checks that the model responds with valid JSON matching the schema
{"selected_option": str, "explanation": str} and, when an expected
option letter is provided, that the model picked the right one.
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
    try:
        data = _extract_json(response)
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Response is not valid JSON: {e}",
            "metadata": {"response_preview": response[:200]},
        }

    if not isinstance(data, dict) or "selected_option" not in data:
        return {
            "passed": False,
            "score": 0.0,
            "reason": "JSON is missing the required 'selected_option' field",
            "metadata": {"parsed": data},
        }

    selected = str(data["selected_option"])

    if expected is None:
        return {
            "passed": True,
            "score": 1.0,
            "reason": "Response is valid JSON with a selected_option",
            "metadata": {"selected_option": selected},
        }

    if _normalize_option(selected) == _normalize_option(expected):
        return {
            "passed": True,
            "score": 1.0,
            "reason": f"Selected option '{selected}' matches expected '{expected}'",
            "metadata": {"selected_option": selected, "explanation": data.get("explanation")},
        }

    return {
        "passed": False,
        "score": 0.0,
        "reason": f"Selected option '{selected}' does not match expected '{expected}'",
        "metadata": {
            "selected_option": selected,
            "expected": expected,
            "explanation": data.get("explanation"),
        },
    }
