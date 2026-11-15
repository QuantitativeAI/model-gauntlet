"""Local model backend (Hugging Face transformers)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

# Intel GPU (oneAPI) tuning knobs — must be set before torch is imported.
os.environ.setdefault("NEOReadDebugKeys", "1")
os.environ.setdefault("ClDeviceOverride", "0")
os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")
os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")
os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .schemas import ModelConfig

DEFAULT_GENERATION_PARAMS: dict[str, Any] = {
    "temperature": 0.3,
    "top_p": 0.85,
    "do_sample": True,
    "max_new_tokens": 400,
    "use_cache": True,
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
        self.device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        # transformers' type stubs are incomplete (missing generate/decode/config/etc.),
        # so these are intentionally left untyped for the checker.
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        print(f"[{self.name}] Loading {self.config.model_id} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        load_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": self.config.device_map or "auto",
        }
        if self.config.max_memory:
            load_kwargs["max_memory"] = self.config.max_memory

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)

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

        messages: dict[str, str] = {}
        if system:
            messages["system"] = system
        messages["user"] = prompt

        generation_params = {**DEFAULT_GENERATION_PARAMS, **self.config.params}
        generation_params["pad_token_id"] = self.tokenizer.eos_token_id
        generation_params["eos_token_id"] = self.tokenizer.eos_token_id
        generation_config = GenerationConfig(**generation_params)

        tokenized = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**tokenized, generation_config=generation_config)
        elapsed = time.time() - start

        input_len = tokenized["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        num_tokens = generated_tokens.numel()
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        tps = num_tokens / elapsed if elapsed > 0 else 0.0
        return GenerationOutput(text=text, elapsed_seconds=elapsed, tokens_per_second=tps, num_tokens=num_tokens)
