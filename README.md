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
other GPU
families (NVIDIA, AMD) are not supported yet — support is planned for a
future version.

## Models

Each model in `config.json` has a `type`:

- `local` (default) — a Hugging Face model id **or a local directory**
  containing `config.json` + `*.safetensors`. Hub ids are downloaded to the
  HF cache on first use and run via transformers.
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
