"""Bootstrap confidence intervals and a paired bootstrap test for comparing two systems."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_test(
    baseline: Sequence[float], candidate: Sequence[float], n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Mean difference (candidate - baseline) and a two-sided paired-bootstrap p-value."""
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test needs scores for the same examples")
    diff = b - a
    observed = float(diff.mean()) if diff.size else 0.0
    if not diff.any():
        return observed, 1.0
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, diff.size, size=(n_resamples, diff.size))].mean(axis=1)
    p = float(np.mean(np.abs(boot - observed) >= abs(observed)))
    return observed, max(p, 1.0 / n_resamples)
