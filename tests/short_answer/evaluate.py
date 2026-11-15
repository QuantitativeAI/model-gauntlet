"""Evaluator for the short_answer suite.

Scores a response by normalized exact match against the expected answer:
case, surrounding whitespace, punctuation and common answer prefixes
("the answer is ...") are ignored. A response that merely contains the
expected answer gets partial credit.
"""

import re

_PREFIX = re.compile(r"^(?:the answer is|answer:|it is|it's)\s+", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = _PREFIX.sub("", text.strip().lower())
    return re.sub(r"[^\w\s]", "", text).strip()


def evaluate(prompt: str, response: str, expected: str | None = None, **kwargs) -> dict:
    """Scores an LLM response against the input prompt and optional expected output."""
    if expected is None:
        return {
            "passed": False,
            "score": 0.0,
            "reason": "short_answer requires an expected value",
            "metadata": {},
        }

    got, want = _normalize(response), _normalize(expected)
    if got == want:
        return {
            "passed": True,
            "score": 1.0,
            "reason": f"Response matches expected answer '{expected}'",
            "metadata": {"response_preview": response[:200]},
        }
    if want and want in got:
        return {
            "passed": False,
            "score": 0.5,
            "reason": f"Expected '{expected}' found inside a longer response",
            "metadata": {"response_preview": response[:200]},
        }
    return {
        "passed": False,
        "score": 0.0,
        "reason": f"Response does not match expected answer '{expected}'",
        "metadata": {"response_preview": response[:200]},
    }
