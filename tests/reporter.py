"""Test suite reporter — generates JSON and HTML reports.

Writes two files per run: a machine-readable JSON report and a
human-readable HTML report with inline styles (no external dependencies).
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from tests.evaluator.behaviour_evaluator import EvalResult


class Reporter:
    """Generates JSON and HTML reports from evaluation results.

    Usage::

        reporter = Reporter(Path("tests/results"))
        reporter.write(eval_results, run_metadata)
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialise the reporter and ensure the output directory exists.

        Args:
            output_dir: Directory where report files will be written.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        results: List[EvalResult],
        run_metadata: Dict,
    ) -> None:
        """Write both JSON and HTML reports.

        Args:
            results: List of EvalResult instances from the test run.
            run_metadata: Dict with keys like tier, runs, timestamp,
                total_cases.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = self._output_dir / f"report_{timestamp}.json"
        html_path = self._output_dir / f"report_{timestamp}.html"

        # -- JSON report ----------------------------------------------------
        json_data = {
            "metadata": run_metadata,
            "results": [asdict(r) for r in results],
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(json_data, fh, indent=2, default=str)

        # -- HTML report ----------------------------------------------------
        html = self._build_html(results, run_metadata)
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)

    # -- Private helpers ----------------------------------------------------

    def _build_html(
        self,
        results: List[EvalResult],
        meta: Dict,
    ) -> str:
        """Build a self-contained HTML report string."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        overall_rate = (passed / total * 100) if total else 0

        # Pass rate by polar_level
        by_polar: Dict[int, List[bool]] = {}
        for r in results:
            by_polar.setdefault(r.polar_level, []).append(r.passed)
        polar_rows = ""
        for pl in sorted(by_polar):
            vals = by_polar[pl]
            rate = sum(vals) / len(vals) * 100
            polar_rows += (
                f'<tr><td style="padding:4px 12px">{pl}</td>'
                f'<td style="padding:4px 12px">{rate:.0f}%</td>'
                f'<td style="padding:4px 12px">{len(vals)}</td></tr>\n'
            )

        # Pass rate by category
        by_cat: Dict[str, List[bool]] = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r.passed)
        cat_rows = ""
        for cat in sorted(by_cat):
            vals = by_cat[cat]
            rate = sum(vals) / len(vals) * 100
            cat_rows += (
                f'<tr><td style="padding:4px 12px">{cat}</td>'
                f'<td style="padding:4px 12px">{rate:.0f}%</td>'
                f'<td style="padding:4px 12px">{len(vals)}</td></tr>\n'
            )

        # Pass rate with modifiers vs without
        with_mod = [r.passed for r in results if r.modifiers]
        without_mod = [r.passed for r in results if not r.modifiers]
        mod_with_rate = (sum(with_mod) / len(with_mod) * 100) if with_mod else 0
        mod_without_rate = (sum(without_mod) / len(without_mod) * 100) if without_mod else 0

        # Full results table
        result_rows = ""
        for r in results:
            if r.verdict == "pass":
                bg = "#d4edda"
            elif r.verdict == "partial":
                bg = "#fff3cd"
            else:
                bg = "#f8d7da"

            mods = ", ".join(r.modifiers) if r.modifiers else "-"
            safety = "PASS" if r.safety_passed else "FAIL"

            result_rows += (
                f'<tr style="background-color:{bg}">'
                f'<td style="padding:4px 8px">{r.case_id}</td>'
                f'<td style="padding:4px 8px;text-align:center">{r.polar_level}</td>'
                f'<td style="padding:4px 8px;text-align:center">{r.category}</td>'
                f'<td style="padding:4px 8px;text-align:center">{r.subtype}</td>'
                f'<td style="padding:4px 8px">{mods}</td>'
                f'<td style="padding:4px 8px;text-align:center">{r.score:.1f}</td>'
                f'<td style="padding:4px 8px;text-align:center">{r.verdict}</td>'
                f'<td style="padding:4px 8px;text-align:center">{safety}</td>'
                f'</tr>\n'
            )

        # Safety violations section
        violation_rows = ""
        for r in results:
            if not r.safety_passed:
                violations_str = "; ".join(r.safety_violations)
                violation_rows += (
                    f'<tr style="background-color:#f8d7da">'
                    f'<td style="padding:4px 8px">{r.case_id}</td>'
                    f'<td style="padding:4px 8px">{violations_str}</td>'
                    f'</tr>\n'
                )

        safety_section = ""
        if violation_rows:
            safety_section = (
                '<h2 style="color:#721c24;margin-top:32px">'
                'Safety Violations</h2>\n'
                '<table style="border-collapse:collapse;width:100%">\n'
                '<tr style="background-color:#721c24;color:white">'
                '<th style="padding:6px 8px;text-align:left">Case</th>'
                '<th style="padding:6px 8px;text-align:left">Violations</th>'
                '</tr>\n'
                f'{violation_rows}'
                '</table>\n'
            )
        else:
            safety_section = (
                '<p style="color:#155724;margin-top:32px;font-weight:bold">'
                'No safety violations detected.</p>\n'
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Antagonistic Robot — Test Suite Report</title></head>
<body style="font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:0 auto;padding:24px;color:#333">

<h1 style="margin-bottom:4px">Antagonistic Robot — Test Suite Report</h1>

<div style="background:#f0f0f0;padding:12px 16px;border-radius:6px;margin-bottom:24px">
  <strong>Tier:</strong> {meta.get("tier", "—")} &nbsp;|&nbsp;
  <strong>Runs per case:</strong> {meta.get("runs", "—")} &nbsp;|&nbsp;
  <strong>Timestamp:</strong> {meta.get("timestamp", "—")} &nbsp;|&nbsp;
  <strong>Total results:</strong> {total}
</div>

<h2>Summary</h2>
<table style="border-collapse:collapse;margin-bottom:16px">
  <tr><td style="padding:4px 12px;font-weight:bold">Overall pass rate</td>
      <td style="padding:4px 12px">{overall_rate:.0f}% ({passed}/{total})</td></tr>
</table>

<div style="display:flex;gap:32px;flex-wrap:wrap">

<div>
<h3>By Polar Level</h3>
<table style="border-collapse:collapse">
<tr style="background:#e9ecef">
  <th style="padding:4px 12px;text-align:left">Polar</th>
  <th style="padding:4px 12px;text-align:left">Pass Rate</th>
  <th style="padding:4px 12px;text-align:left">Count</th></tr>
{polar_rows}</table>
</div>

<div>
<h3>By Category</h3>
<table style="border-collapse:collapse">
<tr style="background:#e9ecef">
  <th style="padding:4px 12px;text-align:left">Cat</th>
  <th style="padding:4px 12px;text-align:left">Pass Rate</th>
  <th style="padding:4px 12px;text-align:left">Count</th></tr>
{cat_rows}</table>
</div>

<div>
<h3>Modifiers</h3>
<table style="border-collapse:collapse">
<tr style="background:#e9ecef">
  <th style="padding:4px 12px;text-align:left">Group</th>
  <th style="padding:4px 12px;text-align:left">Pass Rate</th>
  <th style="padding:4px 12px;text-align:left">Count</th></tr>
<tr><td style="padding:4px 12px">With modifiers</td>
    <td style="padding:4px 12px">{mod_with_rate:.0f}%</td>
    <td style="padding:4px 12px">{len(with_mod)}</td></tr>
<tr><td style="padding:4px 12px">Without modifiers</td>
    <td style="padding:4px 12px">{mod_without_rate:.0f}%</td>
    <td style="padding:4px 12px">{len(without_mod)}</td></tr>
</table>
</div>

</div>

<h2 style="margin-top:32px">Full Results</h2>
<table style="border-collapse:collapse;width:100%">
<tr style="background:#343a40;color:white">
  <th style="padding:6px 8px;text-align:left">case_id</th>
  <th style="padding:6px 8px;text-align:center">polar</th>
  <th style="padding:6px 8px;text-align:center">cat</th>
  <th style="padding:6px 8px;text-align:center">sub</th>
  <th style="padding:6px 8px;text-align:left">modifiers</th>
  <th style="padding:6px 8px;text-align:center">score</th>
  <th style="padding:6px 8px;text-align:center">verdict</th>
  <th style="padding:6px 8px;text-align:center">safety</th>
</tr>
{result_rows}</table>

{safety_section}

</body>
</html>"""

        return html
