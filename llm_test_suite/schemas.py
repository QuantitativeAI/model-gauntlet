"""Pydantic schemas shared across the test runner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationResult(BaseModel):
    """Contract for the dictionary returned by every suite's evaluate()."""

    passed: bool = Field(..., description="Whether the test passed or failed")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0 and 1")
    reason: str = Field(..., description="Explanation for the score/pass status")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional extra data")


class ModelConfig(BaseModel):
    """One entry in the config.json "models" mapping."""

    type: str = Field("local", description="Model backend type (only 'local' is supported in v1)")
    model_id: str = Field(..., description="Hugging Face model id or local path")
    params: dict[str, Any] = Field(default_factory=dict, description="GenerationConfig overrides")
    tests: list[str] = Field(default_factory=list, description="Test suite folder names under tests/")
    device_map: str | None = Field(None, description="Optional accelerate device_map (default: auto)")
    max_memory: dict[str, Any] | None = Field(None, description="Optional accelerate max_memory mapping")


class AppConfig(BaseModel):
    """Top-level config.json schema."""

    models: dict[str, ModelConfig] = Field(default_factory=dict)
    output_dir: str = Field("test_results", description="Results directory, relative to the project root")


class PromptEntry(BaseModel):
    """One row of a suite's prompts/prompts.json.

    Unknown keys are kept and forwarded to evaluate() as **kwargs.
    """

    model_config = ConfigDict(extra="allow")

    prompt: str
    expected: str | None = None
    system: str | None = None

    def extra_kwargs(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class PromptResult(BaseModel):
    """Outcome of a single prompt for a single suite."""

    test: str
    prompt: str
    expected: str | None = None
    response: str
    evaluation: EvaluationResult
    elapsed_seconds: float | None = None
    tokens_per_second: float | None = None


class RunSummary(BaseModel):
    total: int
    passed: int
    pass_rate: float
    avg_score: float


class RunRecord(BaseModel):
    """One row of test_results/<model>/results.json (unique per run)."""

    timestamp: str
    model: str
    model_id: str
    params: dict[str, Any]
    results: list[PromptResult]
    summary: RunSummary
