"""
Evaluation utilities for preference learning sessions.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import spearmanr


def _as_2d(points: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def sample_uniform(bounds: Sequence[Tuple[float, float]], n_samples: int) -> np.ndarray:
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (low, high) pairs.")
    n_samples = max(1, int(n_samples))
    lows = bounds_arr[:, 0]
    highs = bounds_arr[:, 1]
    return np.random.uniform(lows, highs, size=(n_samples, bounds_arr.shape[0]))


def posterior_mean(gp, phys_points: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
    phys = _as_2d(phys_points)
    norm = gp.normalize_parameters(phys)
    mu_vals = gp.mean1pt(norm, eval=True)
    return np.asarray(mu_vals, dtype=float).reshape(-1)


def posterior_variance(gp, phys_points: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
    phys = _as_2d(phys_points)
    norm = gp.normalize_parameters(phys)
    return np.asarray([float(gp.cov1pt(x)) for x in norm], dtype=float)


def posterior_uncertainty(
    gp, bounds: Sequence[Tuple[float, float]], n_samples: int = 500
) -> Dict[str, float]:
    samples = sample_uniform(bounds, n_samples)
    vars_ = posterior_variance(gp, samples)
    return {
        "avg_pred_var": float(np.mean(vars_)) if vars_.size else 0.0,
        "max_pred_var": float(np.max(vars_)) if vars_.size else 0.0,
    }


def correlation_metrics(preds: Sequence[float], gt: Sequence[float]) -> Dict[str, float]:
    preds_arr = np.asarray(preds, dtype=float)
    gt_arr = np.asarray(gt, dtype=float)
    if preds_arr.size == 0 or gt_arr.size == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    if np.std(preds_arr) < 1e-12 or np.std(gt_arr) < 1e-12:
        return {"pearson": 0.0, "spearman": 0.0}
    pearson = float(np.corrcoef(preds_arr, gt_arr)[0, 1])
    sr = spearmanr(preds_arr, gt_arr)
    spearman = float(0.0 if sr.correlation is None or np.isnan(sr.correlation) else sr.correlation)
    return {"pearson": pearson, "spearman": spearman}


def pairwise_accuracy(preds: Sequence[float], gt: Sequence[float], n_pairs: int = 3500) -> float:
    preds_arr = np.asarray(preds, dtype=float)
    gt_arr = np.asarray(gt, dtype=float)
    n = len(gt_arr)
    if n == 0:
        return 0.5
    idx_i = np.random.randint(0, n, size=n_pairs)
    idx_j = np.random.randint(0, n, size=n_pairs)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    gt_diff = gt_arr[idx_i] - gt_arr[idx_j]
    pr_diff = preds_arr[idx_i] - preds_arr[idx_j]
    keep = np.abs(gt_diff) > 1e-12
    if keep.sum() == 0:
        return 0.5
    return float(((gt_diff[keep] > 0) == (pr_diff[keep] > 0)).mean())


def top_k_by_mean(
    gp, bounds: Sequence[Tuple[float, float]], n_samples: int = 5000, k: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    samples = sample_uniform(bounds, n_samples)
    mu_vals = posterior_mean(gp, samples)
    order = np.argsort(mu_vals)[::-1]
    k = max(1, int(k))
    top_idx = order[:k]
    return samples[top_idx], mu_vals[top_idx]


def validation_summary(
    records,
    recommended_params: Optional[np.ndarray],
    competitor_params: Optional[np.ndarray],
) -> Dict[str, object]:
    rounds = len(records)
    wins = sum(1 for rec in records if getattr(rec, "choice", "") == "A")
    win_rate = wins / rounds if rounds else 0.0
    return {
        "rounds": rounds,
        "win_rate": float(win_rate),
        "records": [
            {"round": int(rec.round_index), "choice": rec.choice, "level": int(rec.level)} for rec in records
        ],
        "recommended_params": (
            list(map(float, np.asarray(recommended_params, dtype=float).tolist()))
            if recommended_params is not None
            else None
        ),
        "competitor_params": (
            list(map(float, np.asarray(competitor_params, dtype=float).tolist()))
            if competitor_params is not None
            else None
        ),
    }


def test_metrics(
    gp,
    eval_pts: Sequence[Sequence[float]],
    eval_gt: Sequence[float],
    recommended_params: Optional[np.ndarray],
    gt_eval_fn: Optional[Callable[[Sequence[float]], float]] = None,
) -> Dict[str, Optional[float]]:
    preds = posterior_mean(gp, eval_pts)
    corr = correlation_metrics(preds, eval_gt)
    gt_arr = np.asarray(eval_gt, dtype=float)
    if gt_arr.size == 0:
        return {
            "pearson": corr["pearson"],
            "spearman": corr["spearman"],
            "regret": None,
            "distance_to_optimum": None,
        }

    best_idx = int(np.argmax(gt_arr))
    gt_best = float(gt_arr[best_idx])
    best_params = np.asarray(eval_pts, dtype=float)[best_idx]

    rec_gt = None
    if recommended_params is not None and gt_eval_fn is not None:
        rec_gt = float(gt_eval_fn(recommended_params))

    if rec_gt is None or gt_best <= 1e-12:
        regret = None
    else:
        regret = float((gt_best - rec_gt) / gt_best)

    dist = None
    if recommended_params is not None:
        dist = float(np.linalg.norm(np.asarray(recommended_params, dtype=float) - best_params))

    return {
        "pearson": corr["pearson"],
        "spearman": corr["spearman"],
        "regret": regret,
        "distance_to_optimum": dist,
    }
