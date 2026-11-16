"""Local model backends: Hugging Face transformers and llama-cpp GGUF."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .schemas import ModelConfig

# Intel GPU (oneAPI) tuning knobs — must be set before torch is imported.
_TORCH_ENV_DEFAULTS: dict[str, str] = {
    "NEOReadDebugKeys": "1",
    "ClDeviceOverride": "0",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "ZES_ENABLE_SYSMAN": "1",
    "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
}


def _import_torch_stack():
    """Import torch + transformers, setting the Intel env knobs first.

    Deliberately lazy: the torch XPU stack pulls an older MKL/SYCL runtime
    from the environment, and loading it before a SYCL-built llama-cpp mixes
    MKL versions in one process (undefined mkl_pds_* symbols). GGUF-only runs
    never import torch at all.
    """
    for key, value in _TORCH_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    return torch, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEFAULT_GENERATION_PARAMS: dict[str, Any] = {
    "temperature": 0.3,
    "top_p": 0.85,
    "do_sample": True,
    "max_new_tokens": 400,
    "use_cache": True,
}

GGUF_DEFAULT_PARAMS: dict[str, Any] = {
    "temperature": 0.3,
    "top_p": 0.85,
    "max_tokens": 400,
}


@dataclass
class GenerationOutput:
    text: str
    elapsed_seconds: float
    tokens_per_second: float
    num_tokens: int


class LocalModel:
    """Loads a Hugging Face causal LM once and generates responses for prompts."""

    def __init__(self, name: str, config: ModelConfig):
        self.name = name
        self.config = config
        self.device: str | None = None
        # transformers' type stubs are incomplete (missing generate/decode/config/etc.),
        # so these are intentionally left untyped for the checker.
        self.tokenizer: Any = None
        self.model: Any = None
        self._torch: Any = None
        self._AutoModel: Any = None
        self._AutoTokenizer: Any = None
        self._GenerationConfig: Any = None

    def load(self) -> None:
        self._torch, self._AutoModel, self._AutoTokenizer, self._GenerationConfig = _import_torch_stack()
        torch = self._torch
        self.device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        print(f"[{self.name}] Loading {self.config.model_id} onto {self.device}...")
        self.tokenizer = self._AutoTokenizer.from_pretrained(self.config.model_id)

        load_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": self.config.device_map or "auto",
        }
        if self.config.max_memory:
            load_kwargs["max_memory"] = self.config.max_memory

        self.model = self._AutoModel.from_pretrained(self.config.model_id, **load_kwargs)

        # Some multimodal configs nest vocab_size under text_config.
        if hasattr(self.model.config, "text_config") and not hasattr(self.model.config, "vocab_size"):
            self.model.config.vocab_size = self.model.config.text_config.vocab_size

        print(f"[{self.name}] Model loaded.")

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None

    def generate(self, prompt: str, system: str | None = None) -> GenerationOutput:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded; call load() first")

        # transformers >= 5 requires a list of {"role", "content"} dicts (dict shorthand was removed).
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        generation_params = {**DEFAULT_GENERATION_PARAMS, **self.config.params}
        generation_params["pad_token_id"] = self.tokenizer.eos_token_id
        generation_params["eos_token_id"] = self.tokenizer.eos_token_id
        generation_config = self._GenerationConfig(**generation_params)

        tokenized = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        start = time.time()
        with self._torch.no_grad():
            outputs = self.model.generate(**tokenized, generation_config=generation_config)
        elapsed = time.time() - start

        input_len = tokenized["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        num_tokens = generated_tokens.numel()
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        tps = num_tokens / elapsed if elapsed > 0 else 0.0
        return GenerationOutput(text=text, elapsed_seconds=elapsed, tokens_per_second=tps, num_tokens=num_tokens)


class _NoThinkingHandler:
    """Wraps a chat formatter, injecting enable_thinking=False into the template render."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return self._inner(**kwargs)


class GGUFModel:
    """Loads a GGUF model via llama-cpp-python and generates responses for prompts."""

    def __init__(self, name: str, config: ModelConfig):
        self.name = name
        self.config = config
        self.llm: Any = None

    def load(self) -> None:
        try:
            try:
                from llama_cpp import Llama as LlamaCls  # llama-cpp-python >= 0.3
            except ImportError:
                from llama_cpp import LLM as LlamaCls  # llama-cpp-python < 0.3
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for type='gguf' models but is not installed. "
                "Install it with a SYCL/oneAPI build for Intel GPU offload (the standard "
                "pip wheel is CPU-only)."
            ) from e

        print(f"[{self.name}] Loading GGUF model {self.config.model_id}...")
        # n_gpu_layers=-1 offloads every layer the build supports; CPU-only builds
        # keep everything on the CPU. The chat template is embedded in the GGUF file.
        self.llm = LlamaCls(
            model_path=self.config.model_id, n_ctx=self.config.n_ctx, n_gpu_layers=-1, verbose=False
        )
        self._disable_thinking_if_supported()
        print(f"[{self.name}] Model loaded.")

    def _disable_thinking_if_supported(self) -> None:
        """Thinking models emit a  block before the answer, which breaks
        strict-JSON scoring. If the model's chat template supports the
        `enable_thinking` variable (Qwen3/MiniCPM-style), turn it off so
        answers come back clean — the equivalent of llama.cpp's --jinja mode
        with the thinking branch disabled."""
        template = self.llm.metadata.get("tokenizer.chat_template") or ""
        if "enable_thinking" not in template:
            return
        # _chat_handlers[chat_format] is a closure that forwards **kwargs to
        # the Jinja2 render, so wrapping it lets us inject enable_thinking=False.
        # (Private attribute, but stable across llama-cpp-python 0.3.x.)
        handler = self.llm._chat_handlers.get(self.llm.chat_format)
        if handler is not None:
            self.llm._chat_handlers[self.llm.chat_format] = _NoThinkingHandler(handler)
            print(f"[{self.name}] Thinking disabled via chat template (enable_thinking=false)")

    def unload(self) -> None:
        self.llm = None

    def generate(self, prompt: str, system: str | None = None) -> GenerationOutput:
        if self.llm is None:
            raise RuntimeError("Model is not loaded; call load() first")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {**GGUF_DEFAULT_PARAMS, **self.config.params}
        # Accept the transformers spelling so the same params block works on both backends.
        if "max_new_tokens" in params:
            params["max_tokens"] = params.pop("max_new_tokens")
        # transformers-only knobs have no llama.cpp equivalent; drop them.
        for key in ("do_sample", "use_cache"):
            params.pop(key, None)

        start = time.time()
        out = self.llm.create_chat_completion(messages=messages, **params)
        elapsed = time.time() - start

        text = out["choices"][0]["message"]["content"].strip()
        num_tokens = out.get("usage", {}).get("completion_tokens", 0)
        tps = num_tokens / elapsed if elapsed > 0 else 0.0
        return GenerationOutput(text=text, elapsed_seconds=elapsed, tokens_per_second=tps, num_tokens=num_tokens)


def create_model(name: str, config: ModelConfig) -> LocalModel | GGUFModel:
    if config.type == "local":
        return LocalModel(name, config)
    if config.type == "gguf":
        return GGUFModel(name, config)
    raise ValueError(f"Unsupported model type '{config.type}' (supported: 'local', 'gguf')")
