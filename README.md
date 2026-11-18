# model-gauntlet

Automated LLM test suite to test and compare models against a baseline
=======

Automated testing of LLM models.

## Layout

```
llm_test_suite/          # entry point, config.json, .env
tests/<suite>/           # one folder per test suite
  prompts/<set>/         # one subfolder per prompt set (*.json, *.jsonl, *.csv or prompts.py)
  evaluate.py            # evaluate(prompt, response, expected=None, **kwargs) -> dict
test_results/<model>/results.json   # one row per run (timestamp + model params)
```

## Setup

```bash
pip install -r requirements.txt
cp llm_test_suite/.env.example llm_test_suite/.env   # add HF_TOKEN etc.
```

## Hardware

The runner targets **Intel GPUs** via the PyTorch XPU backend (oneAPI). On
start it checks `torch.xpu.is_available()` and uses the XPU when present;
otherwise it **falls back to CPU** (works, but much slower). Other GPU
families (NVIDIA, AMD) are not supported yet — support is planned for a
future version.

## Models

Each model in `config.json` has a `type`:

- `transformers` (default) — a Hugging Face model id **or a local directory**
  containing `config.json` + `*.safetensors`. Hub ids are downloaded to the
  HF cache on first use.
- `gguf` — a path to a `.gguf` file, run via
  [llama-cpp-python](https://github.com/leloykun/llama-cpp-python) with the
  chat template embedded in the file. `model_id` must point at the file.

`gguf` is an optional dependency: you only need llama-cpp-python if you
configure a `type: "gguf"` model.

**llama-cpp-python must be built locally from source.** A plain
`pip install llama-cpp-python` fetches a prebuilt wheel that is CPU-only and
may not even match your Python version. For Intel GPU offload, build it
against the oneAPI SYCL toolchain:

```bash
# oneAPI environment is required so icx/icpx are on PATH:
source /opt/intel/oneapi/setvars.sh
CMAKE_ARGS="-DGGML_SYCL=on -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \
    FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

`FORCE_CMAKE=1` skips prebuilt wheels and compiles llama.cpp from source —
expect several minutes. Use the `pip` of the Python environment that runs
the app.

**One more step: link the oneAPI runtime libraries.** The build only embeds
the MKL library path, so `libllama.so` still can't find a handful of oneAPI
runtime libs (`libiomp5.so`, `libsvml.so`, `libsycl.so.9`, ...). Symlink them
from your oneAPI install into the package's `lib/` folder:

```bash
LIBDIR=$(python -c "import llama_cpp, os; print(os.path.dirname(llama_cpp.__file__))")/lib
while :; do
    missing=$(ldd "$LIBDIR/libllama.so" 2>/dev/null | awk '/not found/{print $1}' | sort -u)
    [ -z "$missing" ] && break
    linked=0
    for m in $missing; do
        src=$(find /opt/intel/oneapi -path "*/lib/$m" 2>/dev/null | grep -v lib32 | head -1)
        if [ -n "$src" ]; then ln -sf "$src" "$LIBDIR/$m"; linked=1; else echo "locate manually: $m"; fi
    done
    [ "$linked" = 0 ] && break
done
```

Then verify in a **fresh shell** (no setvars):

```bash
python -c "from llama_cpp import Llama; print('llama-cpp import OK')"
```

Do **not** run the app in a shell where `setvars.sh` is sourced: its
`LD_LIBRARY_PATH` shadows the SYCL runtime bundled with the torch wheel and
breaks `import torch`. Build in one shell, run the app in a fresh one.

Generation params merge over per-backend defaults; for `gguf` models the
transformers spelling `max_new_tokens` is accepted and translated to llama.cpp's
`max_tokens`. `n_ctx` (default 4096) sets the GGUF context window.

`seed` (default 0, `gguf` only) fixes the sampling seed so runs are
reproducible: with the same seed, params and prompt order, a rerun produces
the same generations, so score changes between runs reflect parameter changes
alone. Set a different value to draw a different (but still reproducible)
sample. The seed is recorded in each run's results row and shown in the
report's Params column. (`transformers` runs are not seeded; very long
generations can also vary slightly between runs due to GPU float
non-determinism.)

`llm_options` (`gguf` only) passes extra options straight to the llama-cpp
`Llama()` constructor — the equivalent of llama-server's load-time flags:

```json
"qwen-27b": {
  "type": "gguf",
  "model_id": "/mnt/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q6_K_L.gguf",
  "n_ctx": 250000,
  "tensor_split": [1.0, 1.0],
  "llm_options": {
    "flash_attn": true,
    "use_mmap": false,
    "type_k": "q4_0",
    "type_v": "q4_0"
  }
}
```

`type_k`/`type_v` accept a type name (`"q4_0"`, `"q8_0"`, `"f16"`, ...) or the
raw llama.cpp enum int. Sampling flags (`temperature`, `top_p`, `top_k`,
`min_p`, `presence_penalty`, `repeat_penalty`, ...) go in `params` as before.
Some llama-server flags have no equivalent here: `--parallel` (prompts run
sequentially through one context), `--jinja` (always on), `--lazy-mode`
(not in the Python API), `--spec-type` (speculative decoding), and
`--reasoning` (thinking is controlled by the chat template instead).

### Multi-GPU (Intel)

Both backends can spread one model across several Intel GPUs. This is for
*fitting* models that don't fit on a single GPU, not a speedup — a split
model pays a cross-device transfer at every layer boundary, so a model that
fits on one GPU runs best on that one alone.

- `transformers` — `"device_map": "auto"` shards layers across all visible XPUs;
  `max_memory` caps the memory per device index:

  ```json
  "big-model": {
    "type": "transformers",
    "model_id": "org/model-70b",
    "device_map": "auto",
    "max_memory": { "0": "28GiB", "1": "21GiB" }
  }
  ```

  A model that fits on one GPU should be **pinned** there instead
  (`"device_map": "xpu:1"`) — sharding adds a cross-device transfer at every
  layer boundary, and multi-device execution on torch XPU is less robust.

- `gguf` — `tensor_split` holds one weight ratio per GPU; `main_gpu`
  (default 0) picks the device that also hosts the non-splittable parts;
  optional `split_mode` is `"row"` (split each tensor across GPUs) or
  `"layer"` (whole layers per GPU). The llama.cpp SYCL backend currently
  only supports `"layer"` — `"row"` aborts at load time, so use `"layer"":

  ```json
  "big-model": {
    "type": "gguf",
    "model_id": "/mnt/models/Model-70B-Q8_0.gguf",
    "tensor_split": [1.0, 1.0],
    "main_gpu": 1
  }
  ```

Device numbers are in SYCL/XPU order; list them with:

```bash
python -c "import torch; print([torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])"
```

The `tests` list assigns suites to a model. Omit the key and the model runs
every suite under `tests/` (new suites are picked up automatically); an
explicit `"tests": []` runs nothing.

## Usage

```bash
# Check config, model availability and suite contracts (no model loading)
python -m llm_test_suite validate

# Run all configured models
python -m llm_test_suite run

# Run a single model
python -m llm_test_suite run --model llama-3.1-8b

# Summarize all recorded runs (markdown by default, or --format html)
python -m llm_test_suite report
python -m llm_test_suite report --format html
```

`validate` and `run` accept `--config PATH` to use an alternative config.
Keep **one backend type per config file**: loading both a transformers
(torch) model and a llama-cpp (gguf) model in one process crashes with an
MKL runtime conflict. The defaults are:

- `llm_test_suite/config.json` — gguf models
- `llm_test_suite/config-safetensors.json` — safetensors/transformers models

```bash
python -m llm_test_suite run --config llm_test_suite/config-safetensors.json
```

Both write into the same `test_results/` directory, so the report covers
runs from all configs.

The report is written to `test_results/report.md` (or `report.html`) and
contains a per-model summary plus one row per run × suite with the model
parameters, date/time, pass rate and average score. Use `--output` to write
it elsewhere.

## Writing a test suite

Create `tests/<name>/` with an `evaluate.py` and a `prompts/` folder. Each
prompt set is its own subfolder of `prompts/` — when a new set of prompts is
added, just create a new subfolder (no config change needed):

```
tests/bar-exams/
  prompts/
    json-format/        # set 1 (exists)
      prompts.json
    multiple-choice/    # set 2 (hypothetical — create when you add it)
      qa.jsonl
  evaluate.py
```

Each set folder contains either a `prompts.py` module or any combination of
`*.json`, `*.jsonl` and `*.csv` files (concatenated in filename order):

- `*.json` — a JSON list of entries
- `*.jsonl` — one JSON entry per line (blank lines are skipped)
- `*.csv` — header row with a required `prompt` column plus optional `expected`
  and `system` columns; empty `expected`/`system` cells mean "none"
- `prompts.py` — exposes a `PROMPTS` list of entry dicts; useful for
  programmatically generated prompts:

```python
PROMPTS = [
    {"prompt": f"What is {a} + {b}?", "expected": str(a + b)}
    for a, b in ((i, i * 2) for i in range(1, 51))
]
```

Each entry needs `prompt` (str) and may include `expected` (str or null) and
`system` (str, sent as a system message). Any extra keys (or CSV columns) are
passed to `evaluate()` as keyword arguments.

When designing a JSON response schema, put free-text fields (e.g.
`explanation`) **before** the final answer field (e.g. `selected_option`).
Small models commit to the answer before they have reasoned, so
answer-first schemas systematically understate them — measured here: 5/10 →
10/10 on a 2B model with identical weights and prompts.

The runner executes every set of every assigned suite, and each result row
records the suite and set it came from.

`evaluate.py` must define a function with the exact contract:

```python
def evaluate(prompt: str, response: str, expected: str | None = None, **kwargs) -> dict:
    return {"passed": True, "score": 1.0, "reason": "...", "metadata": {}}
```

Then assign the suite to a model in `llm_test_suite/config.json` under
`"tests"` — or leave the key out of a model entry, in which case that model
runs every suite.
