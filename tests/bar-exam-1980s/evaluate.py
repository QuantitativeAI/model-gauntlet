"""Evaluator for the bar-exams suite.

Scores multiple-choice law questions. Full credit requires a valid JSON
response ({"selected_option": ..., "explanation": ...}) with the correct
option; a correct answer given without the JSON format earns partial credit.
"""

import json
import re


def _repair_backslashes(text: str) -> str:
    """Double backslashes that are not valid JSON escapes (e.g. LaTeX \\sin).

    Small models often emit LaTeX with single backslashes, which JSON rejects;
    repairing them lets an otherwise well-formed response parse. Backslashes
    already part of a valid escape (e.g. a doubled ``\\``) are left alone, so
    mixed-escaping responses (``\\int`` next to ``\\,``) still parse.
    """
    valid_escapes = '"\\/bfnrtu'
    hex_digits = "0123456789abcdefABCDEF"
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt in valid_escapes:
            end = i + 2
            if nxt == "u":
                while end < n and end - i < 6 and text[end] in hex_digits:
                    end += 1
            out.append(text[i:end])
            i = end
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


def _loads_lenient(text: str):
    """json.loads, retrying once with repaired backslash escapes."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_backslashes(text))


def _extract_json(text: str):
    """Parse JSON from a response, tolerating code fences and surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return _loads_lenient(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return _loads_lenient(text[start : end + 1])
        raise


def _normalize_option(value: str) -> str:
    """Reduce an option to its letter: 'B) Something...' -> 'B', 'option b' -> 'B'."""
    value = value.strip().upper()
    match = re.match(r"^(?:OPTION\s+)?([A-Z])(?=[\s).:]|$)", value)
    return match.group(1) if match else value


def _option_texts(prompt: str) -> dict[str, str]:
    """Map option letter -> option text as listed in the prompt (e.g. 'A) Duty of care')."""
    return {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"^\s*([A-Z])\)\s*(.+?)\s*$", prompt, re.MULTILINE)
    }


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


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
        correct = _normalize_option(selected) == _normalize_option(expected)
        if not correct:
            # The model may echo the option text without its letter; accept an
            # exact match against the expected option's text as listed in the prompt.
            expected_text = _option_texts(prompt).get(expected.strip().upper())
            correct = bool(expected_text) and _norm_text(selected) == _norm_text(expected_text)
        if correct:
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
