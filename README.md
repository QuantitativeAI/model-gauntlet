# model-gauntlet

Automated LLM test suite to test and compare models against a baseline
=======

Automated testing of LLM models.

## Layout

```
llm_test_suite/          # entry point, config.json, .env
tests/<suite>/           # one folder per test suite
  prompts/               # *.json and/or *.jsonl and/or *.csv prompt files
  prompts.py             # alternative to prompts/: module exposing PROMPTS = [...]
  evaluate.py            # evaluate(prompt, response, expected=None, **kwargs) -> dict
test_results/<model>/results.json   # one row per run (timestamp + model params)
```

## Setup

```bash
pip install -r requirements.txt
cp llm_test_suite/.env.example llm_test_suite/.env   # add HF_TOKEN etc.
```

## Usage

```bash
# Check config, model availability and suite contracts (no model loading)
python -m llm_test_suite validate

# Run all configured models
python -m llm_test_suite run

# Run a single model
python -m llm_test_suite run --model llama-3.1-8b
```

## Writing a test suite

Create `tests/<name>/` with an `evaluate.py` and a prompts source — either a
`prompts/` folder or a `prompts.py` module (not both):

**`prompts/` folder** — any combination of `*.json`, `*.jsonl` and `*.csv` files,
concatenated in filename order:

- `*.json` — a JSON list of entries
- `*.jsonl` — one JSON entry per line (blank lines are skipped)
- `*.csv` — header row with a required `prompt` column plus optional `expected`
  and `system` columns; empty `expected`/`system` cells mean "none"

**`prompts.py` module** — exposes a `PROMPTS` list of entry dicts; useful for
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

`evaluate.py` must define a function with the exact contract:

```python
def evaluate(prompt: str, response: str, expected: str | None = None, **kwargs) -> dict:
    return {"passed": True, "score": 1.0, "reason": "...", "metadata": {}}
```

Then assign the suite to a model in `llm_test_suite/config.json` under `"tests"`.
