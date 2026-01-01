"""
Evaluation utilities for preference learning sessions.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize
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


def find_strong_ground_truth(
    gt_func: Callable[[Sequence[float]], float],
    bounds: Sequence[Tuple[float, float]],
    n_random: int = 10000,
    top_k: int = 20,
) -> Tuple[float, np.ndarray]:
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (low, high) pairs.")
    samples = sample_uniform(bounds, n_random)
    gt_vals = []
    for x in samples:
        try:
            val = float(gt_func(x))
        except Exception:
            val = float("-inf")
        gt_vals.append(val)
    gt_vals = np.asarray(gt_vals, dtype=float)
    gt_vals = np.where(np.isfinite(gt_vals), gt_vals, -np.inf)

    if gt_vals.size == 0:
        raise ValueError("No samples available for ground-truth search.")

    order = np.argsort(gt_vals)[::-1]
    k = max(1, min(int(top_k), len(order)))
    top_idx = order[:k]

    best_val = float(gt_vals[top_idx[0]])
    best_params = np.asarray(samples[top_idx[0]], dtype=float)

    bounds_list = [(float(lo), float(hi)) for lo, hi in bounds_arr]

    def negative_gt(x: np.ndarray) -> float:
        return -float(gt_func(x))

    for idx in top_idx:
        x0 = np.asarray(samples[idx], dtype=float)
        try:
            res = minimize(negative_gt, x0=x0, bounds=bounds_list, method="L-BFGS-B")
            val = -float(res.fun)
            if np.isfinite(val) and val > best_val:
                best_val = val
                best_params = np.asarray(res.x, dtype=float)
        except Exception:
            continue

    return float(best_val), np.asarray(best_params, dtype=float)


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
    true_optimum_params: Sequence[float],
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

    regret = None
    if gt_eval_fn is not None and recommended_params is not None and true_optimum_params is not None:
        gt_best = float(gt_eval_fn(true_optimum_params))
        gt_rec = float(gt_eval_fn(recommended_params))
        regret = float(max(0.0, (gt_best - gt_rec) / (abs(gt_best) + 1e-12)))

    dist = None
    if recommended_params is not None and true_optimum_params is not None:
        dist = float(
            np.linalg.norm(np.asarray(recommended_params, dtype=float) - np.asarray(true_optimum_params, dtype=float))
        )

    return {
        "pearson": corr["pearson"],
        "spearman": corr["spearman"],
        "regret": regret,
        "distance_to_optimum": dist,
    }


def recursive_sanitize(obj):
    if isinstance(obj, dict):
        return {k: recursive_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [recursive_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return recursive_sanitize(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj
