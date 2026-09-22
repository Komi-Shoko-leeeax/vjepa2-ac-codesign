
from __future__ import annotations

from typing import Dict
import numpy as np
from scipy.stats import kendalltau, spearmanr


def topk_indices(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x)
    k = min(int(k), x.size)
    return np.argpartition(x, kth=k - 1)[:k]


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ia = set(topk_indices(a, k).tolist())
    ib = set(topk_indices(b, k).tolist())
    if not ia:
        return 1.0
    return len(ia & ib) / len(ia)


def pairwise_ranking_flip_rate(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Fraction of candidate pairs whose relative ordering flips.
    O(N^2); use on analysis subsets for very large N.
    """
    r = np.asarray(reference).reshape(-1)
    t = np.asarray(test).reshape(-1)
    if r.shape != t.shape:
        raise ValueError("reference/test energy arrays must have the same shape")
    n = len(r)
    if n < 2:
        return 0.0
    flips = 0
    valid = 0
    for i in range(n):
        dr = r[i] - r[i + 1 :]
        dt = t[i] - t[i + 1 :]
        mask = (dr != 0) & (dt != 0)
        valid += int(mask.sum())
        flips += int(((np.sign(dr[mask]) != np.sign(dt[mask]))).sum())
    return float(flips / valid) if valid else 0.0


def compare_rankings(
    reference: np.ndarray,
    test: np.ndarray,
    topk: int = 10,
) -> Dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    test = np.asarray(test, dtype=np.float64).reshape(-1)
    if reference.shape != test.shape:
        raise ValueError("reference/test arrays must have the same shape")

    sp = spearmanr(reference, test)
    kt = kendalltau(reference, test)

    ref_best = int(np.argmin(reference))
    test_best = int(np.argmin(test))

    abs_err = np.abs(test - reference)
    denom = np.maximum(np.abs(reference), 1e-12)

    return {
        "spearman": float(sp.statistic),
        "kendall_tau": float(kt.statistic),
        "topk_overlap": float(topk_overlap(reference, test, topk)),
        "argmin_consistency": float(ref_best == test_best),
        "reference_best_idx": ref_best,
        "test_best_idx": test_best,
        "energy_mae": float(abs_err.mean()),
        "energy_relative_mae": float((abs_err / denom).mean()),
        "pairwise_flip_rate": float(pairwise_ranking_flip_rate(reference, test)),
    }
