"""CI quality gate over result JSON files, using dotted metric paths.

thresholds.yaml::

    security:                                   # result file key (see `sqlagent eval gate`)
      attack_block_rate.combined: {min: 1.0}
      benign_false_block_rate.guard: {max: 0.0}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateFailure:
    result: str
    metric: str
    value: Any
    rule: str

    def __str__(self) -> str:
        return f"{self.result}: {self.metric} = {self.value} (required {self.rule})"


def _lookup(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def check(results: dict[str, Any], thresholds: dict[str, dict[str, dict]]) -> list[GateFailure]:
    failures = []
    for result_name, metrics in thresholds.items():
        for metric, rule in metrics.items():
            value = _lookup(results.get(result_name), metric)
            ok = isinstance(value, int | float | bool)
            if ok and "min" in rule:
                ok = value >= rule["min"]
            if ok and "max" in rule:
                ok = value <= rule["max"]
            if not ok:
                desc = ", ".join(f"{k} {v}" for k, v in rule.items())
                failures.append(GateFailure(result_name, metric, value, desc))
    return failures
