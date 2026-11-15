"""Pre-run validation: config, model availability, suite layout and contracts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import evaluators
from .schemas import AppConfig, ModelConfig


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    subject: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(config: AppConfig, tests_dir: Path) -> ValidationReport:
    report = ValidationReport()

    if not config.models:
        report.issues.append(Issue("error", "config", "No models defined in config.json"))
    if not tests_dir.is_dir():
        report.issues.append(Issue("error", "tests", f"Tests directory not found: {tests_dir}"))
        return report

    referenced: set[str] = set()
    download_needed = False
    for model_name, model_cfg in config.models.items():
        referenced.update(model_cfg.tests)
        download_needed |= _validate_model(report, model_name, model_cfg, tests_dir)

    _check_hf_token(report, download_needed)

    for folder in sorted(tests_dir.iterdir()):
        if folder.is_dir():
            _validate_suite(report, folder, referenced=folder.name in referenced)

    return report


def _validate_model(report: ValidationReport, name: str, cfg: ModelConfig, tests_dir: Path) -> bool:
    """Validate one model. Returns True if the model must be downloaded from Hugging Face."""
    if cfg.type != "local":
        report.issues.append(
            Issue("error", name, f"Unsupported model type '{cfg.type}' (only 'local' is supported in v1)")
        )
        return False
    if not cfg.model_id:
        report.issues.append(Issue("error", name, "model_id is empty"))
        return False

    needs_download = not (Path(cfg.model_id).is_dir() or _in_hf_cache(cfg.model_id))
    if needs_download:
        report.issues.append(
            Issue(
                "warning",
                name,
                f"'{cfg.model_id}' not found in the local cache; it will be downloaded on first run "
                "(requires network access and, for gated models, HF_TOKEN in .env)",
            )
        )

    if not cfg.tests:
        report.issues.append(Issue("warning", name, "No test suites assigned to this model"))
    for test in cfg.tests:
        if not (tests_dir / test).is_dir():
            report.issues.append(Issue("error", name, f"Assigned test suite '{test}' not found under {tests_dir}"))

    return needs_download


def _check_hf_token(report: ValidationReport, download_needed: bool) -> None:
    if not download_needed:
        return
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        report.issues.append(
            Issue("warning", "env", "HF_TOKEN is not set; gated models (e.g. Llama) cannot be downloaded")
        )
        return
    if not token.isascii():
        report.issues.append(
            Issue(
                "error",
                "env",
                "HF_TOKEN contains non-ASCII characters (e.g. a '…' copied from documentation); "
                "set your real token from https://huggingface.co/settings/tokens",
            )
        )
    elif "your-token" in token.lower():
        report.issues.append(
            Issue(
                "error",
                "env",
                "HF_TOKEN still contains the placeholder from .env.example; "
                "set your real token from https://huggingface.co/settings/tokens",
            )
        )


def _validate_suite(report: ValidationReport, folder: Path, referenced: bool) -> None:
    name = folder.name
    try:
        evaluators.load_evaluator(folder)
    except evaluators.EvaluatorContractError as e:
        report.issues.append(Issue("error", name, str(e)))
    except Exception as e:  # noqa: BLE001 - report any suite error (e.g. syntax error in evaluate.py)
        report.issues.append(Issue("error", name, f"Unexpected error validating suite: {e}"))

    try:
        set_names = evaluators.list_prompt_sets(folder)
    except evaluators.EvaluatorContractError as e:
        report.issues.append(Issue("error", name, str(e)))
    else:
        for set_name in set_names:
            try:
                evaluators.load_prompt_set(folder, set_name)
            except evaluators.EvaluatorContractError as e:
                report.issues.append(Issue("error", f"{name}/{set_name}", str(e)))
            except Exception as e:  # noqa: BLE001 - report any set error (e.g. syntax error in prompts.py)
                report.issues.append(Issue("error", f"{name}/{set_name}", f"Unexpected error: {e}"))

    if not referenced:
        report.issues.append(Issue("warning", name, "Suite exists but is not assigned to any model in config.json"))


def _in_hf_cache(model_id: str) -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return (hf_home / "hub" / f"models--{model_id.replace('/', '--')}").is_dir()
