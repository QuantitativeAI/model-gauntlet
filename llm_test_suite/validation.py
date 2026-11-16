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
    runs_all = False
    download_needed = False
    for model_name, model_cfg in config.models.items():
        if model_cfg.tests is None:
            runs_all = True  # omitted 'tests' means every suite is assigned
        else:
            referenced.update(model_cfg.tests)
        download_needed |= _validate_model(report, model_name, model_cfg, tests_dir)

    _check_hf_token(report, download_needed)

    for folder in sorted(tests_dir.iterdir()):
        if folder.is_dir():
            _validate_suite(report, folder, referenced=runs_all or folder.name in referenced)

    return report


def _validate_model(report: ValidationReport, name: str, cfg: ModelConfig, tests_dir: Path) -> bool:
    """Validate one model. Returns True if the model must be downloaded from Hugging Face."""
    if cfg.type not in ("local", "gguf"):
        report.issues.append(
            Issue("error", name, f"Unsupported model type '{cfg.type}' (supported: 'local', 'gguf')")
        )
        return False
    if not cfg.model_id:
        report.issues.append(Issue("error", name, "model_id is empty"))
        return False

    if cfg.split_mode not in (None, "row", "layer"):
        report.issues.append(Issue("error", name, f"Unsupported split_mode '{cfg.split_mode}' (supported: 'row', 'layer')"))
    if cfg.tensor_split is not None and any(v <= 0 for v in cfg.tensor_split):
        report.issues.append(Issue("error", name, "tensor_split values must be positive ratios"))
    if cfg.type == "local" and (cfg.tensor_split is not None or cfg.split_mode is not None):
        report.issues.append(
            Issue("warning", name, "tensor_split/split_mode are ignored for type 'local' (use device_map/max_memory instead)")
        )

    if cfg.type == "gguf":
        needs_download = False
        if not Path(cfg.model_id).is_file():
            report.issues.append(Issue("error", name, f"GGUF model file not found: {cfg.model_id}"))
        elif not _llama_cpp_available():
            report.issues.append(
                Issue(
                    "warning",
                    name,
                    "type 'gguf' requires llama-cpp-python, which is not installed; "
                    "use a SYCL/oneAPI build for Intel GPU offload",
                )
            )
    else:
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

    if cfg.tests is not None and not cfg.tests:
        report.issues.append(
            Issue("warning", name, "'tests' is empty; this model will run no suites (omit the key to run all)")
        )
    for test in cfg.tests or []:
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


def _llama_cpp_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None
