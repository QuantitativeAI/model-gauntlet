"""Report writer: summarizes all recorded runs as markdown or HTML."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class ReportRow:
    """One report row: a single test suite within a single run."""

    timestamp: str
    model: str
    params: str
    suite: str
    total: int
    passed: int
    pass_rate: float
    avg_score: float


def collect_rows(results_dir: Path) -> list[ReportRow]:
    """Read every test_results/<model>/results.json and flatten to report rows."""
    rows: list[ReportRow] = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        results_file = model_dir / "results.json"
        if not results_file.is_file():
            continue
        try:
            records = json.loads(results_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ! Skipping {results_file}: not valid JSON")
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                rows.extend(_rows_for_record(record))
    rows.sort(key=lambda r: (r.model, r.timestamp))
    return rows


def _rows_for_record(record: dict) -> list[ReportRow]:
    model = str(record.get("model", "?"))
    timestamp = str(record.get("timestamp", ""))
    params = format_params(record.get("params") or {})
    seed = record.get("seed")
    if seed is not None:
        params = f"{params} seed={seed}" if params else f"seed={seed}"

    by_suite: dict[str, list[dict]] = {}
    for result in record.get("results", []):
        if isinstance(result, dict):
            by_suite.setdefault(str(result.get("test", "?")), []).append(result)

    rows = []
    for suite in sorted(by_suite):
        entries = by_suite[suite]
        total = len(entries)
        passed = sum(1 for e in entries if (e.get("evaluation") or {}).get("passed"))
        scores = [(e.get("evaluation") or {}).get("score", 0.0) for e in entries]
        rows.append(
            ReportRow(
                timestamp=timestamp,
                model=model,
                params=params,
                suite=suite,
                total=total,
                passed=passed,
                pass_rate=passed / total if total else 0.0,
                avg_score=sum(scores) / total if total else 0.0,
            )
        )
    return rows


def format_params(params: dict) -> str:
    if not params:
        return "defaults"
    aliases = {"temperature": "temp"}
    parts = []
    for k, v in params.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        parts.append(f"{aliases.get(k, k)}={v}")
    return " ".join(parts)


def _display_timestamp(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def _model_summaries(rows: list[ReportRow]) -> list[dict]:
    groups: dict[str, list[ReportRow]] = {}
    for row in rows:
        groups.setdefault(row.model, []).append(row)

    summaries = []
    for model, model_rows in groups.items():
        total = sum(r.total for r in model_rows)
        passed = sum(r.passed for r in model_rows)
        summaries.append(
            {
                "model": model,
                "runs": len({r.timestamp for r in model_rows}),
                "total": total,
                "passed": passed,
                "pass_rate": passed / total if total else 0.0,
                "avg_score": sum(r.avg_score * r.total for r in model_rows) / total if total else 0.0,
                "latest": max(r.timestamp for r in model_rows),
            }
        )
    return summaries


def render_markdown(rows: list[ReportRow]) -> str:
    lines = [
        "# model-gauntlet — Test Results Report",
        "",
        f"_Generated: {datetime.now(UTC).isoformat(timespec='seconds')}_",
        "",
        "## Summary by Model",
        "",
        "| Model | Runs | Prompts | Passed | Pass % | Avg Score | Latest Run |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in _model_summaries(rows):
        lines.append(
            f"| {s['model']} | {s['runs']} | {s['total']} | {s['passed']} "
            f"| {s['pass_rate']:.0%} | {s['avg_score']:.1%} | {_display_timestamp(s['latest'])} |"
        )
    lines += [
        "",
        "## Runs by Suite",
        "",
        "| Date/Time | Model | Params | Suite | Prompts | Passed | Pass % | Avg Score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {_display_timestamp(r.timestamp)} | {r.model} | {r.params} | {r.suite} "
            f"| {r.total} | {r.passed} | {r.pass_rate:.0%} | {r.avg_score:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_html(rows: list[ReportRow]) -> str:
    e = html.escape
    generated = e(datetime.now(UTC).isoformat(timespec="seconds"))

    summary_rows = "\n".join(
        "<tr>"
        f"<td>{e(s['model'])}</td><td>{s['runs']}</td><td>{s['total']}</td><td>{s['passed']}</td>"
        f"<td>{s['pass_rate']:.0%}</td><td>{s['avg_score']:.1%}</td>"
        f"<td>{e(_display_timestamp(s['latest']))}</td>"
        "</tr>"
        for s in _model_summaries(rows)
    )

    run_rows = "\n".join(
        "<tr>"
        f"<td>{e(_display_timestamp(r.timestamp))}</td><td>{e(r.model)}</td><td>{e(r.params)}</td>"
        f"<td>{e(r.suite)}</td><td>{r.total}</td><td>{r.passed}</td>"
        f"<td class=\"{'pass' if r.pass_rate >= 0.8 else 'fail' if r.pass_rate < 0.5 else ''}\">{r.pass_rate:.0%}</td>"
        f"<td>{r.avg_score:.1%}</td>"
        "</tr>"
        for r in rows
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>model-gauntlet — Test Results Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; font-size: 0.9rem; }}
th {{ background: #f0f0f0; }}
tr:nth-child(even) td {{ background: #fafafa; }}
.pass {{ color: #1a7f37; font-weight: 600; }}
.fail {{ color: #cf222e; font-weight: 600; }}
</style>
</head>
<body>
<h1>model-gauntlet — Test Results Report</h1>
<p>Generated: {generated}</p>
<h2>Summary by Model</h2>
<table>
<tr><th>Model</th><th>Runs</th><th>Prompts</th><th>Passed</th><th>Pass %</th><th>Avg Score</th><th>Latest Run</th></tr>
{summary_rows}
</table>
<h2>Runs by Suite</h2>
<table>
<tr><th>Date/Time</th><th>Model</th><th>Params</th><th>Suite</th><th>Prompts</th><th>Passed</th><th>Pass %</th><th>Avg Score</th></tr>
{run_rows}
</table>
</body>
</html>
"""
