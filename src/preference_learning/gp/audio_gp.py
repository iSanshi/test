"""
Audio-specific Gaussian Process helpers for four-dimensional preference learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .gaussian_process import GaussianProcess


ParameterBounds = Dict[str, Tuple[float, float]]


@dataclass
class AudioPreferenceGaussianProcess(GaussianProcess):
    """Gaussian Process with fixed 4D parameter bounds for audio optimization."""

    parameter_bounds: ParameterBounds = field(
        default_factory=lambda: {
            "amplitude": (20.0, 100.0),
            "frequency": (20.0, 100.0),
            "density": (20.0, 100.0),
            "gradient": (20.0, 100.0),
        }
    )

    def __post_init__(self) -> None:
        if self.initial_point is None:
            self.initial_point = [0.5, 0.5, 0.5, 0.5]
        super().__post_init__()
        if self.dim != 4:
            raise ValueError("AudioPreferenceGaussianProcess expects a 4D initial point.")

    # ------------------------------------------------------------------ #
    # Normalisation helpers
    # ------------------------------------------------------------------ #
    def normalize_parameters(self, params: Sequence[float]) -> np.ndarray:
        """Normalize physical parameters to [0, 1]."""
        params_arr = np.asarray(params, dtype=float)
        bounds = np.array(list(self.parameter_bounds.values()), dtype=float)
        normalized = (params_arr - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        return np.clip(normalized, 0.0, 1.0)

    def denormalize_parameters(self, normalized_params: Sequence[float]) -> np.ndarray:
        """Convert normalized parameters back to physical ranges."""
        normalized = np.asarray(normalized_params, dtype=float)
        bounds = np.array(list(self.parameter_bounds.values()), dtype=float)
        return normalized * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    # ------------------------------------------------------------------ #
    # Query optimisation
    # ------------------------------------------------------------------ #
    def find_optimal_query(self, n_restarts: int = 5) -> Tuple[np.ndarray, float]:
        """Return the best next query (two 4D points) and its information gain."""

        def negative_info_gain(x: np.ndarray) -> float:
            return -1.0 * self.objective_entropy(x)

        bounds = [(0.0, 1.0)] * 8
        best_result = None
        best_info_gain = -np.inf

        for _ in range(max(n_restarts, 1)):
            base_point = np.concatenate([self.initialPoint, self.initialPoint])
            perturbation = np.random.uniform(-0.3, 0.3, 8)
            start = np.clip(base_point + perturbation, 0.0, 1.0)

            try:
                opt_res = minimize(
                    negative_info_gain,
                    x0=start,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"ftol": 1e-9, "gtol": 1e-6},
                )
                info_gain = -opt_res.fun
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_result = opt_res.x
            except Exception:
                continue

        if best_result is None:
            start = np.random.uniform(0.0, 1.0, 8)
            opt_res = minimize(negative_info_gain, x0=start, bounds=bounds, method="L-BFGS-B")
            best_result = opt_res.x
            best_info_gain = -opt_res.fun

        return best_result, best_info_gain

    # Backwards-compatible alias
    find_optimal_query_4d = find_optimal_query
