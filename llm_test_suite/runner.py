"""Test execution: model -> suite -> prompt."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from . import evaluators
from .config import AppConfig
from .schemas import (
    EvaluationResult,
    ModelConfig,
    PromptEntry,
    PromptResult,
    RunRecord,
    RunSummary,
)


def run_tests(
    config: AppConfig,
    tests_dir: Path,
    output_dir: Path,
    model_names: list[str] | None = None,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for model_name, model_cfg in config.models.items():
        if model_names and model_name not in model_names:
            continue
        records.append(_run_model(model_name, model_cfg, tests_dir, output_dir))
    return records


def _run_model(model_name: str, model_cfg: ModelConfig, tests_dir: Path, output_dir: Path) -> RunRecord:
    # Imported lazily so that `validate` works without torch/transformers installed.
    from .model_backend import LocalModel

    model = LocalModel(model_name, model_cfg)
    model.load()
    prompt_results: list[PromptResult] = []
    try:
        for test_name in model_cfg.tests:
            test_dir = tests_dir / test_name
            evaluator = evaluators.load_evaluator(test_dir)
            for set_name in evaluators.list_prompt_sets(test_dir):
                entries = evaluators.load_prompt_set(test_dir, set_name)
                print(
                    f"[{model_name}] Running suite '{test_name}' / set '{set_name}' "
                    f"({len(entries)} prompts)..."
                )
                for entry in entries:
                    prompt_results.append(_run_prompt(test_name, set_name, entry, model, evaluator))
    finally:
        model.unload()

    summary = _summarize(prompt_results)
    record = RunRecord(
        timestamp=datetime.now(UTC).isoformat(),
        model=model_name,
        model_id=model_cfg.model_id,
        params=dict(model_cfg.params),
        results=prompt_results,
        summary=summary,
    )
    _append_run(output_dir / model_name / "results.json", record)
    _print_summary(model_name, summary)
    return record


def _run_prompt(test_name: str, set_name: str, entry: PromptEntry, model, evaluator) -> PromptResult:
    extras = entry.extra_kwargs()
    try:
        gen = model.generate(entry.prompt, system=entry.system)
        response = gen.text
        elapsed, tps = gen.elapsed_seconds, gen.tokens_per_second
    except Exception as e:  # noqa: BLE001 - isolate per-prompt generation failures
        return PromptResult(
            test=test_name,
            set=set_name,
            prompt=entry.prompt,
            expected=entry.expected,
            response="",
            evaluation=EvaluationResult(passed=False, score=0.0, reason=f"Generation failed: {e}", metadata={}),
        )

    raw = None
    try:
        raw = evaluator(prompt=entry.prompt, response=response, expected=entry.expected, **extras)
        evaluation = EvaluationResult.model_validate(raw)
    except Exception as e:  # noqa: BLE001 - isolate per-prompt evaluation failures
        evaluation = EvaluationResult(
            passed=False,
            score=0.0,
            reason=f"Evaluation failed: {e}",
            metadata={"raw_return": str(raw)[:500]},
        )

    status = "PASS" if evaluation.passed else "FAIL"
    print(f"  [{status}] score={evaluation.score:.2f}  {evaluation.reason[:100]}")
    return PromptResult(
        test=test_name,
        set=set_name,
        prompt=entry.prompt,
        expected=entry.expected,
        response=response,
        evaluation=evaluation,
        elapsed_seconds=elapsed,
        tokens_per_second=tps,
    )


def _summarize(results: list[PromptResult]) -> RunSummary:
    total = len(results)
    passed = sum(1 for r in results if r.evaluation.passed)
    avg_score = sum(r.evaluation.score for r in results) / total if total else 0.0
    return RunSummary(
        total=total,
        passed=passed,
        pass_rate=passed / total if total else 0.0,
        avg_score=avg_score,
    )


def _append_run(path: Path, record: RunRecord) -> None:
    existing: list = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            print(f"  ! Existing {path} is not valid JSON; starting a new file")
    existing.append(record.model_dump(mode="json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"  -> Results written to {path}")


def _print_summary(model_name: str, summary: RunSummary) -> None:
    print(
        f"[{model_name}] Done: {summary.passed}/{summary.total} passed "
        f"(pass rate {summary.pass_rate:.0%}, avg score {summary.avg_score:.2f})"
    )
