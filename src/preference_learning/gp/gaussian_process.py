"""
Gaussian Process implementation tailored for pairwise preference learning.

Refactored from the original `GP_ours.py` implementation while keeping the
behaviour and public API intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b, minimize
from scipy.stats import multivariate_normal

from .math_utils import (
    h,
    normal_cdf,
    phi,
    phip,
    phipp,
)


UncertaintySigmaDict = Dict[int, float]
PreferenceDict = Dict[Tuple[float, ...], float]
Query = List[np.ndarray]


def _as_array(point: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert input to a one-dimensional numpy array."""
    arr = np.asarray(point, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Points must be one-dimensional sequences.")
    return arr


@dataclass
class GaussianProcess:
    """Gaussian Process model for pairwise preference learning."""

    initial_point: Optional[Union[Sequence[float], np.ndarray]] = None
    theta: float = 0.1
    noise_level: float = 0.1
    
    uncertainty_sigma_dict: UncertaintySigmaDict = field(
        default_factory=lambda: {1: 0.01, 2: 0.66, 3: 1.7, 4: 3.35, 5: 9.0}

    )


    def __post_init__(self) -> None:
        if self.initial_point is None:
            raise ValueError("initial_point must be provided.")
        self.noise: float = self.noise_level
        self.initialPoint: np.ndarray = _as_array(self.initial_point)
        self.dim: int = len(self.initialPoint)

        self.listQueries: List[List] = []
        self.pref_dict: PreferenceDict = {}
        self.uncertainty_level: int = 0

        # Internal GP state matrices
        self.K: np.ndarray = np.zeros((2, 2))
        self.Kinv: np.ndarray = np.zeros((2, 2))
        self.fqmean: np.ndarray = np.zeros(2)
        self.W: np.ndarray = np.zeros((2, 2))

    # ------------------------------------------------------------------ #
    # Core GP update helpers
    # ------------------------------------------------------------------ #
    def update_parameters(
        self,
        query: Sequence[Sequence[float]],
        answer: int,
        uncertainty: int,
        pref_dict: PreferenceDict,
    ) -> None:
        """Update model parameters with a new query result."""
        xa = _as_array(query[0])
        xb = _as_array(query[1])
        self.listQueries.append([xa, xb, answer, uncertainty])
        self.uncertainty_level = uncertainty
        self.pref_dict = pref_dict

        self.K = self._covariance_full()
        identity = np.identity(2 * len(self.listQueries))
        self.Kinv = inv(self.K + identity * 1e-8)
        self.fqmean = self._posterior_mode()
        self.W = self._hessian()

    # Keep original method name for backwards compatibility
    updateParameters = update_parameters

    # ------------------------------------------------------------------ #
    # Information gain objective
    # ------------------------------------------------------------------ #
    def objective_entropy(self, concatenated_points: np.ndarray) -> float:
        """Compute the information gain objective for a pair of points."""
        xa = concatenated_points[: self.dim]
        xb = concatenated_points[self.dim :]

        matCov = self.postcov(xa, xb)
        mua, mub = self.postmean(xa, xb)
        sigmap = np.sqrt(np.pi * np.log(2) / 2) * self.noise

        result1 = h(
            phi(
                (mua - mub)
                / np.sqrt(
                    2 * self.noise**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]
                )
            )
        )
        result2 = (
            sigmap
            * 1
            / np.sqrt(sigmap**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1])
            * np.exp(
                -0.5
                * (mua - mub) ** 2
                / (sigmap**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1])
            )
        )

        return result1 - result2

    objectiveEntropy = objective_entropy

    # ------------------------------------------------------------------ #
    # Query optimization (shared with auto-test UI)
    # ------------------------------------------------------------------ #
    def find_optimal_query(self, n_restarts: int = 5) -> Tuple[np.ndarray, float]:
        """Return the best next query (two points) and its information gain."""

        def negative_info_gain(x: np.ndarray) -> float:
            return -1 * self.objective_entropy(x)

        bounds = [(0.0, 1.0)] * (2 * self.dim)
        best_result = None
        best_info_gain = -np.inf

        for _ in range(max(n_restarts, 1)):
            base_point = np.array(list(self.initialPoint) * 2)
            lower = np.zeros_like(base_point)
            upper = np.ones_like(base_point)
            start = base_point + np.random.uniform(lower - base_point, upper - base_point)

            try:
                opt_res = fmin_l_bfgs_b(
                    negative_info_gain,
                    x0=start,
                    bounds=bounds,
                    approx_grad=True,
                    factr=0.1,
                    iprint=-1,
                )
                info_gain = -opt_res[1]
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_result = opt_res[0]
            except Exception:
                continue

        if best_result is None:
            start = np.random.uniform(0.0, 1.0, 2 * self.dim)
            opt_res = fmin_l_bfgs_b(
                negative_info_gain,
                x0=start,
                bounds=bounds,
                approx_grad=True,
                factr=0.1,
                iprint=-1,
            )
            best_result = opt_res[0]
            best_info_gain = -opt_res[1]

        return np.asarray(best_result, dtype=float), float(best_info_gain)

    # ------------------------------------------------------------------ #
    # Posterior calculations
    # ------------------------------------------------------------------ #
    def _gmm_weights(self, xa: np.ndarray, xb: np.ndarray) -> Tuple[float, float]:
        total_pdf_a = 0.0
        total_pdf_b = 0.0
        covariance = 1 / (np.sqrt(2 * np.pi))

        for pref, count in self.pref_dict.items():
            rv = multivariate_normal(np.array(pref), covariance)
            total_pdf_a += count * rv.pdf(xa)
            total_pdf_b += count * rv.pdf(xb)

        total_pdf_a += 1
        total_pdf_b += 1
        return 1 / total_pdf_a, 1 / total_pdf_b

    GMM = _gmm_weights

    def kernel(
        self,
        xa: Union[Sequence[float], np.ndarray],
        xb: Union[Sequence[float], np.ndarray],
    ) -> float:
        """Squared exponential kernel."""
        ker = 1 * (np.exp(-self.theta * np.linalg.norm(np.array(xa) - np.array(xb)) ** 2))
        try:
            ker = ker[0]
        except Exception:
            pass
        if ker < 0:
            print("You can not have a negative kernel!")
            raise SystemExit(1)
        return ker

    def batch_kernel(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        """Batch kernel evaluation for a matrix of points."""
        num = xa.shape[0]
        xb = np.repeat(np.array(xb).reshape(1, -1), num, axis=0)
        return np.exp(-self.theta * np.linalg.norm(np.array(xa) - np.array(xb), axis=1) ** 2)

    def _posterior_mode(self) -> np.ndarray:
        """Find posterior means for the queries."""
        num_queries = len(self.listQueries)
        Kinv = self.Kinv
        listResults = np.array([q[2] for q in self.listQueries])
        sigmas = np.array([self.uncertainty_sigma_dict[q[3]] for q in self.listQueries])

        def logposterior(f: np.ndarray) -> float:
            fodd = f[1::2]
            feven = f[::2]
            fint = 1 / self.noise * (feven - fodd)
            res = np.multiply(fint, listResults)
            res = res.astype(dtype=np.float64)
            res = normal_cdf(res, scale=sigmas[:num_queries])
            res[res == 0] = 1e-100
            res = np.log(res)
            res = np.sum(res)
            ftransp = f.reshape(-1, 1)
            return -1 * (res - 0.5 * np.matmul(f, np.matmul(Kinv, ftransp)))

        def gradientlog(f: np.ndarray) -> np.ndarray:
            grad = np.zeros(2 * len(self.listQueries))
            for i in range(len(self.listQueries)):
                signe = self.listQueries[i][2]
                diff = f[2 * i] - f[2 * i + 1]
                temp = phi(signe * 1 / self.noise * (diff), sigma=sigmas[i])
                if temp == 0:
                    temp = 1e-100
                grad[2 * i] = (
                    signe
                    * (phip(signe * 1 / self.noise * (diff), sigma=sigmas[i]) * 1 / self.noise)
                    / temp
                )
                grad[2 * i + 1] = (
                    signe
                    * (-phip(signe * 1 / self.noise * (diff), sigma=sigmas[i]) * 1 / self.noise)
                    / temp
                )
            grad = grad - f @ Kinv
            return -grad

        x0 = np.zeros(2 * num_queries)
        return minimize(logposterior, x0=x0, jac=gradientlog).x

    meanmode = _posterior_mode

    def _hessian(self) -> np.ndarray:
        """Compute the Hessian matrix."""
        n = len(self.listQueries)
        W = np.zeros((2 * n, 2 * n))
        for i in range(n):
            dif = (
                self.listQueries[i][2]
                * 1
                / self.noise
                * (self.fqmean[2 * i] - self.fqmean[2 * i + 1])
            )
            W[2 * i][2 * i] = (
                -(1 / self.noise**2)
                * (phipp(dif) * phi(dif) - phip(dif) ** 2)
                / (phi(dif) ** 2)
            )
            W[2 * i + 1][2 * i] = -W[2 * i][2 * i]
            W[2 * i][2 * i + 1] = -W[2 * i][2 * i]
            W[2 * i + 1][2 * i + 1] = W[2 * i][2 * i]
        return W

    hessian = _hessian

    def kt(self, xa: np.ndarray, xb: np.ndarray, eval: bool = False) -> np.ndarray:
        """Compute covariance between new points and existing queries."""
        n = len(self.listQueries)
        if eval:
            return np.array(
                [self.batch_kernel(xa, self.listQueries[i][j]) for i in range(n) for j in range(2)]
            )

        return np.array(
            [
                [self.kernel(xa, self.listQueries[i][j]) for i in range(n) for j in range(2)],
                [self.kernel(xb, self.listQueries[i][j]) for i in range(n) for j in range(2)],
            ]
        )

    def _covariance_full(self) -> np.ndarray:
        n = len(self.listQueries)
        return np.array(
            [
                [
                    self.kernel(self.listQueries[i][j], self.listQueries[l][m])
                    for l in range(n)
                    for m in range(2)
                ]
                for i in range(n)
                for j in range(2)
            ]
        )

    covK = _covariance_full

    def postmean(self, xa: np.ndarray, xb: np.ndarray, eval: bool = False) -> np.ndarray:
        kt = self.kt(xa, xb, eval=eval)
        if eval:
            kt = kt.T
        return np.matmul(kt, np.matmul(self.Kinv, self.fqmean))

    def postcov(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        n = len(self.listQueries)
        Kt = np.array(
            [
                [self.kernel(xa, xa), self.kernel(xa, xb)],
                [self.kernel(xb, xa), self.kernel(xb, xb)],
            ]
        )
        kt = self.kt(xa, xb)
        W = self.W
        K = self.K
        post_cov = Kt - kt @ inv(np.identity(2 * n) + np.matmul(W, K)) @ W @ np.transpose(kt)

        xaa, xbb = self._gmm_weights(xa, xb)
        post_cov[0][0] *= xaa**2
        post_cov[0][1] *= xaa * xbb
        post_cov[1][0] *= xaa * xbb
        post_cov[1][1] *= xbb**2
        return post_cov

    def cov1pt(self, x: np.ndarray) -> float:
        return self.postcov(x, 0)[0][0]

    def mean1pt(self, x: np.ndarray, eval: bool = False) -> Union[float, np.ndarray]:
        if eval:
            return self.postmean(x, 0, eval=True)
        return self.postmean(x, 0)[0]
