"""Markdown tables for agent, linking and security results."""

from __future__ import annotations

from typing import Any

from sqlagent.eval.execution import FAILURE_CATEGORIES
from sqlagent.eval.stats import paired_bootstrap_test


def agent_table(runs: list[dict[str, Any]]) -> str:
    base = runs[0]
    lines = [
        "| System | EX | Δ vs `" + base["name"] + "` | easy / medium / hard | $/task "
        "| tool calls | p50 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        ex = r["execution_accuracy"]
        if r is base:
            delta = "—"
        else:
            d, p = paired_bootstrap_test(base["per_example"], r["per_example"])
            delta = f"{d:+.3f} ({'p<0.001' if p < 0.001 else f'p={p:.3f}'})" + (
                " ✱" if p < 0.05 else "")
        diff = " / ".join(f"{r['by_difficulty'][d]['mean']:.2f}"
                          for d in ("easy", "medium", "hard") if d in r["by_difficulty"])
        lines.append(
            f"| `{r['name']}` | {ex['mean']:.3f} <sub>[{ex['ci_low']:.2f}, {ex['ci_high']:.2f}]"
            f"</sub> | {delta} | {diff} | ${r['cost_per_task_usd']:.4f} "
            f"| {r['mean_tool_calls']:.1f} | {r['latency_ms']['p50'] / 1000:.1f} s |"
        )
    lines += ["", "| System | " + " | ".join(FAILURE_CATEGORIES) + " |",
              "|---|" + "---|" * len(FAILURE_CATEGORIES)]
    for r in runs:
        lines.append(f"| `{r['name']}` | "
                     + " | ".join(str(r["failures"][c]) for c in FAILURE_CATEGORIES) + " |")
    return "\n".join(lines)


def linking_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| k | FK expansion | all gold tables kept | table recall | tables shown "
             "| schema size vs full |", "|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["covered_ci"]
        lines.append(
            f"| {r['k']} | {'yes' if r['fk_expansion'] else 'no'} "
            f"| {r['all_gold_tables_covered']:.3f} <sub>[{lo:.2f}, {hi:.2f}]</sub> "
            f"| {r['table_recall']:.3f} | {r['mean_tables_shown']:.1f} "
            f"| {r['schema_size_vs_full']:.0%} |"
        )
    return "\n".join(lines)


def security_table(s: dict[str, Any]) -> str:
    lines = ["| Attack category | n | blocked by guard | blocked by sandbox (guard bypassed) "
             "| blocked by either |", "|---|---|---|---|---|"]
    for cat, c in sorted(s["by_category"].items()):
        lines.append(f"| {cat} | {c['n']} | {c['guard']}/{c['n']} | {c['sandbox']}/{c['n']} "
                     f"| {c['both']}/{c['n']} |")
    a, b = s["attack_block_rate"], s["benign_false_block_rate"]
    lines.append(f"| **all attacks** | {s['n_attacks']} | {a['guard']:.0%} | {a['sandbox']:.0%} "
                 f"| **{a['combined']:.0%}** |")
    lines += [
        "",
        f"Benign queries wrongly blocked ({s['n_benign']}, including every gold query): "
        f"guard {b['guard']:.0%}, sandbox {b['sandbox']:.0%}. "
        "Database file unchanged after all attacks: "
        f"**{'yes' if s['database_unchanged'] else 'NO'}**.",
    ]
    return "\n".join(lines)
