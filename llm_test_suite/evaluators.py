"""Loading and contract validation of test suite evaluators."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import PromptEntry


class EvaluatorContractError(ValueError):
    """Raised when a suite's evaluate.py or prompts.json does not match the expected contract."""


def load_evaluator(test_dir: Path) -> Callable:
    """Import <test_dir>/evaluate.py and return its validated evaluate() function."""
    module_path = test_dir / "evaluate.py"
    if not module_path.is_file():
        raise EvaluatorContractError(f"Missing evaluate.py in {test_dir}")

    module_name = f"model_gauntlet_evaluator_{test_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise EvaluatorContractError(f"Could not create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    func = getattr(module, "evaluate", None)
    if func is None or not callable(func):
        raise EvaluatorContractError(f"'evaluate' not found or is not callable in {module_path}")

    _validate_signature(func, module_path)
    return func


def _validate_signature(func: Callable, module_path: Path) -> None:
    sig = inspect.signature(func)
    params = sig.parameters

    for name in ("prompt", "response"):
        param = params.get(name)
        if param is None:
            raise EvaluatorContractError(f"{module_path}: evaluate() is missing required parameter '{name}'")
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise EvaluatorContractError(f"{module_path}: evaluate() parameter '{name}' must be a named parameter")

    expected = params.get("expected")
    if expected is not None and expected.default is inspect.Parameter.empty:
        raise EvaluatorContractError(
            f"{module_path}: evaluate() parameter 'expected' must be optional (give it a default value)"
        )

    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        raise EvaluatorContractError(f"{module_path}: evaluate() must accept **kwargs for extra prompt fields")

    ret = sig.return_annotation
    if ret is not inspect.Signature.empty and ret is not dict:
        raise EvaluatorContractError(f"{module_path}: evaluate() must return dict, but is annotated as {ret!r}")


PROMPT_FILE_SUFFIXES = (".json", ".jsonl", ".csv")


def load_prompts(test_dir: Path) -> list[PromptEntry]:
    """Load the prompt entries for a suite.

    Supported sources (exactly one must be present):
      1. <test_dir>/prompts.py  - a module exposing PROMPTS: a list of entry dicts
      2. <test_dir>/prompts/    - any *.json, *.jsonl and *.csv files,
                                  concatenated in filename order
    """
    py_source = test_dir / "prompts.py"
    data_dir = test_dir / "prompts"
    if py_source.is_file() and data_dir.is_dir():
        raise EvaluatorContractError(
            f"{test_dir}: both prompts.py and prompts/ found; keep only one prompts source"
        )
    if py_source.is_file():
        return _load_prompts_py(py_source)
    if data_dir.is_dir():
        return _load_prompts_dir(data_dir)
    raise EvaluatorContractError(
        f"Missing prompts source in {test_dir}: expected prompts.py or prompts/ "
        "containing *.json, *.jsonl or *.csv files"
    )


def _load_prompts_dir(data_dir: Path) -> list[PromptEntry]:
    files = sorted(
        p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in PROMPT_FILE_SUFFIXES
    )
    if not files:
        raise EvaluatorContractError(
            f"No prompt files found in {data_dir} (expected *.json, *.jsonl or *.csv)"
        )
    entries: list[PromptEntry] = []
    for path in files:
        entries.extend(_load_prompt_file(path))
    return entries


def _load_prompt_file(path: Path) -> list[PromptEntry]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = _read_json_list(path)
        return [_validate_row(path, f"row {i}", row) for i, row in enumerate(rows, start=1)]
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".csv":
        return _read_csv(path)
    raise EvaluatorContractError(f"{path}: unsupported prompt file type '{suffix}'")


def _read_json_list(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise EvaluatorContractError(f"{path}: expected a JSON list of prompt entries")
    return data


def _read_jsonl(path: Path) -> list[PromptEntry]:
    entries: list[PromptEntry] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise EvaluatorContractError(f"{path}: line {lineno} is not valid JSON: {e}") from e
        entries.append(_validate_row(path, f"line {lineno}", row))
    return entries


def _read_csv(path: Path) -> list[PromptEntry]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "prompt" not in (reader.fieldnames or []):
            raise EvaluatorContractError(f"{path}: missing required 'prompt' column")
        entries: list[PromptEntry] = []
        for lineno, row in enumerate(reader, start=2):  # line 1 is the header
            entry: dict[str, Any] = {}
            for key, value in row.items():
                if key is None or value is None:
                    continue  # stray positional values / missing cells
                value = value.strip()
                if key in ("expected", "system"):
                    entry[key] = value or None
                elif key == "prompt":
                    entry[key] = value
                elif value:
                    entry[key] = value  # extra columns are forwarded to evaluate()
            if not entry.get("prompt"):
                raise EvaluatorContractError(f"{path}: line {lineno} has an empty 'prompt'")
            entries.append(_validate_row(path, f"line {lineno}", entry))
    return entries


def _load_prompts_py(path: Path) -> list[PromptEntry]:
    module_name = f"model_gauntlet_prompts_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvaluatorContractError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = getattr(module, "PROMPTS", None)
    if rows is None:
        raise EvaluatorContractError(f"{path} must expose a 'PROMPTS' list of prompt entries")
    if not isinstance(rows, list):
        raise EvaluatorContractError(f"{path}: 'PROMPTS' must be a list of prompt entries")
    return [_validate_row(path, f"row {i}", row) for i, row in enumerate(rows, start=1)]


def _validate_row(path: Path, label: str, row: Any) -> PromptEntry:
    try:
        return PromptEntry.model_validate(row)
    except ValidationError as e:
        raise EvaluatorContractError(f"{path}: {label} is invalid: {e}") from e
