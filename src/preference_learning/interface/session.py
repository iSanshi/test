"""
Session/business logic for the audio preference learning application.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from ..audio import AudioGenerator
from ..gp import AudioPreferenceGaussianProcess


LEVEL_TO_WEIGHT = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
DIFF_THRESHOLDS = [0.01, 0.05, 0.10, 0.20]


def level_from_diff_legacy(diff_abs_norm: float) -> int:
    level = 1
    for j, threshold in enumerate(DIFF_THRESHOLDS, start=1):
        if diff_abs_norm > threshold:
            level = j + 1
    return max(1, min(5, level))


def moving_average(arr: Sequence[float], k: int = 5) -> np.ndarray:
    if len(arr) == 0:
        return np.array([])
    k = max(1, int(k))
    weights = np.ones(k) / k
    return np.convolve(np.asarray(arr, dtype=float), weights, mode="valid")


class SessionMode(Enum):
    USER = "User Study"
    TEST = "Test"


class GroundTruthKind(Enum):
    GAUSSIAN_CENTER = "Gaussian (center)"
    GAUSSIAN_OFFSET = "Gaussian (offset)"
    BIMODAL = "Bimodal (2 peaks)"
    RIDGE = "Ridge (anisotropic)"

    @classmethod
    def from_label(cls, label: str) -> "GroundTruthKind":
        for item in cls:
            if item.value == label:
                return item
        return cls.GAUSSIAN_CENTER


@dataclass
class AudioCandidate:
    normalized: Tuple[np.ndarray, np.ndarray]
    physical: Tuple[np.ndarray, np.ndarray]
    info_gain: float
    query_distance: float
    audio_data: Dict[int, Dict[str, np.ndarray]]


@dataclass
class TestQueryRecord:
    iteration: int
    physical: Tuple[np.ndarray, np.ndarray]
    choice: str
    level: int
    info_gain: float
    query_distance: float


@dataclass
class Recommendation:
    parameters: Optional[np.ndarray] = None
    mean_value: Optional[float] = None


@dataclass
class SessionState:
    mode: SessionMode = SessionMode.USER
    max_iterations: int = 40
    current_iteration: int = 0
    running: bool = False
    gt_kind: GroundTruthKind = GroundTruthKind.GAUSSIAN_CENTER


class PreferenceSession:
    """Encapsulates the non-UI logic of the audio preference learning workflow."""

    def __init__(self, audio_generator: Optional[AudioGenerator] = None) -> None:
        self.audio = audio_generator or AudioGenerator()
        self.gp: Optional[AudioPreferenceGaussianProcess] = None
        self.state = SessionState()

        self.pref_dict: Dict[Tuple[float, ...], float] = {}
        self.current_candidates: Optional[AudioCandidate] = None
        self.rec_best = Recommendation()
        self.ideal_phys = np.array([60.0, 60.0, 60.0, 60.0])

        # Histories
        self.preference_history: List[int] = []
        self.parameter_history_phys: List[np.ndarray] = []
        self.uncertainty_history: List[int] = []
        self.info_gain_history: List[float] = []
        self.query_distance_history: List[float] = []
        self.level_count_history: List[List[int]] = []
        self.pearson_history: List[float] = []
        self.spearman_history: List[float] = []
        self.pairacc_history: List[float] = []
        self.best_so_far_history: List[float] = []
        self.rec_distance_history: List[float] = []
        self.rec_history_phys: List[np.ndarray] = []
        self.test_query_history: List[TestQueryRecord] = []

        # Test evaluation
        self.eval_pts_phys: Optional[List[np.ndarray]] = None
        self.eval_gt: Optional[np.ndarray] = None

        self._test_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._stop_event.set()
        if self._test_thread and self._test_thread.is_alive():
            self._test_thread.join(timeout=0.5)
        self._stop_event.clear()

        self.gp = None
        self.pref_dict.clear()
        self.current_candidates = None
        self.rec_best = Recommendation()

        self.preference_history.clear()
        self.parameter_history_phys.clear()
        self.uncertainty_history.clear()
        self.info_gain_history.clear()
        self.query_distance_history.clear()
        self.level_count_history.clear()
        self.pearson_history.clear()
        self.spearman_history.clear()
        self.pairacc_history.clear()
        self.best_so_far_history.clear()
        self.rec_distance_history.clear()
        self.rec_history_phys.clear()
        self.test_query_history.clear()
        self.eval_pts_phys = None
        self.eval_gt = None

        self.state.current_iteration = 0
        self.state.running = False

    def start(self, mode: SessionMode, max_iterations: int, gt_label: str) -> None:
        self.reset()
        self.state.mode = mode
        self.state.max_iterations = max(1, max_iterations)
        self.state.gt_kind = GroundTruthKind.from_label(gt_label)

        initial_point = [0.5, 0.5, 0.5, 0.5]
        self.gp = AudioPreferenceGaussianProcess(initial_point=initial_point, theta=0.5, noise_level=0.1)

        center = np.array(initial_point)
        self.pref_dict = {tuple(center): LEVEL_TO_WEIGHT[3]}
        self.gp.update_parameters([center, center], 0, 3, self.pref_dict)

        self.state.running = True
        self.ideal_phys = np.array(self._ideal_for_gt_func(self.state.gt_kind), dtype=float)

        if self.state.mode is SessionMode.TEST:
            self._build_eval_set(n=800, seed=123)

    def stop(self) -> None:
        self.state.running = False
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # User mode helpers
    # ------------------------------------------------------------------ #
    def generate_user_query(self) -> Optional[AudioCandidate]:
        if not self.state.running or self.state.mode is not SessionMode.USER or self.gp is None:
            return None

        query, info_gain = self.gp.find_optimal_query()
        p1_norm = np.asarray(query[:4])
        p2_norm = np.asarray(query[4:])
        p1_phys = self.gp.denormalize_parameters(p1_norm)
        p2_phys = self.gp.denormalize_parameters(p2_norm)

        dist = float(np.linalg.norm(p1_norm - p2_norm))

        audio_data = {}
        for idx, params in enumerate([p1_phys, p2_phys], start=1):
            t, x, meta = self.audio.generate_signal(*params)
            plot_waveform = meta.get("for_plot", {})
            if isinstance(plot_waveform, dict) and "plot_waveform" in plot_waveform:
                plot_vals = np.asarray(plot_waveform["plot_waveform"])
            else:
                plot_vals = x
            if plot_vals.ndim == 2:
                plot_vals = np.mean(plot_vals, axis=1)
            audio_data[idx] = {"t": t, "x": x, "meta": meta, "plot": plot_vals}

        candidate = AudioCandidate(
            normalized=(p1_norm, p2_norm),
            physical=(p1_phys, p2_phys),
            info_gain=float(info_gain),
            query_distance=dist,
            audio_data=audio_data,
        )
        self.current_candidates = candidate
        self.info_gain_history.append(float(info_gain))
        self.query_distance_history.append(dist)
        return candidate

    def record_user_choice(self, choice_label: str, level: int) -> None:
        if not self.current_candidates or self.gp is None:
            return

        level = int(np.clip(level, 1, 5))
        y = 1 if choice_label == "A" else -1
        p1_norm, p2_norm = self.current_candidates.normalized
        p1_phys, p2_phys = self.current_candidates.physical

        for key in (tuple(p1_norm), tuple(p2_norm)):
            self.pref_dict[key] = self.pref_dict.get(key, 0.0) + LEVEL_TO_WEIGHT[level]
        self.gp.update_parameters([p1_norm, p2_norm], y, level, self.pref_dict)
        if level == 1:
            self.gp.update_parameters([p1_norm, p2_norm], -y, level, self.pref_dict)

        chosen_phys = p1_phys if y == 1 else p2_phys
        self.parameter_history_phys.append(chosen_phys)
        self.preference_history.append(y)
        self.uncertainty_history.append(level)

        counts = np.bincount(np.array(self.uncertainty_history, dtype=int), minlength=6)[1:6]
        self.level_count_history.append(counts.tolist())

        self.info_gain_history.append(self.current_candidates.info_gain)
        self.query_distance_history.append(self.current_candidates.query_distance)

        self.state.current_iteration += 1
        self._update_recommendation()

    # ------------------------------------------------------------------ #
    # Test mode helpers
    # ------------------------------------------------------------------ #
    def run_test_loop(self) -> None:
        if self._test_thread and self._test_thread.is_alive():
            return

        self._stop_event.clear()
        self._test_thread = threading.Thread(target=self._test_worker, daemon=True)
        self._test_thread.start()

    def _test_worker(self) -> None:
        while (
            not self._stop_event.is_set()
            and self.state.running
            and self.state.current_iteration < self.state.max_iterations
            and self.gp is not None
        ):
            try:
                query, info_gain = self.gp.find_optimal_query()
                p1_norm = np.asarray(query[:4])
                p2_norm = np.asarray(query[4:])
                p1_phys = self.gp.denormalize_parameters(p1_norm)
                p2_phys = self.gp.denormalize_parameters(p2_norm)

                gt1 = self._gt_value(p1_phys)
                gt2 = self._gt_value(p2_phys)

                if self.eval_gt is None:
                    self._build_eval_set(n=800, seed=123)
                gt_min = float(np.min(self.eval_gt))
                gt_max = float(np.max(self.eval_gt))
                gt_range = max(gt_max - gt_min, 1e-12)
                diff_norm = abs(gt1 - gt2) / gt_range
                level = level_from_diff_legacy(diff_norm)

                y = 1 if gt1 > gt2 else -1
                for key in (tuple(p1_norm), tuple(p2_norm)):
                    self.pref_dict[key] = self.pref_dict.get(key, 0.0) + LEVEL_TO_WEIGHT[level]
                self.gp.update_parameters([p1_norm, p2_norm], y, level, self.pref_dict)
                if level == 1:
                    self.gp.update_parameters([p1_norm, p2_norm], -y, level, self.pref_dict)

                chosen_phys = p1_phys if y == 1 else p2_phys
                self.parameter_history_phys.append(chosen_phys)
                self.preference_history.append(y)
                self.uncertainty_history.append(level)

                counts = np.bincount(np.array(self.uncertainty_history, dtype=int), minlength=6)[1:6]
                self.level_count_history.append(counts.tolist())

                self.info_gain_history.append(float(info_gain))
                qdist = float(np.linalg.norm(p1_norm - p2_norm))
                self.query_distance_history.append(qdist)

                self.state.current_iteration += 1
                self.test_query_history.append(
                    TestQueryRecord(
                        iteration=self.state.current_iteration,
                        physical=(np.asarray(p1_phys, dtype=float).copy(), np.asarray(p2_phys, dtype=float).copy()),
                        choice=("A" if y == 1 else "B"),
                        level=int(level),
                        info_gain=float(info_gain),
                        query_distance=qdist,
                    )
                )
                self._update_test_metrics(update_rec_best=True)
            except Exception:
                break

            time.sleep(0.2)

        self.state.running = False

    # ------------------------------------------------------------------ #
    # Metrics and evaluation
    # ------------------------------------------------------------------ #
    def _update_recommendation(self) -> None:
        if not self.parameter_history_phys:
            return
        selected = np.array(self.parameter_history_phys[-1], dtype=float)
        self.rec_best = Recommendation(parameters=selected, mean_value=None)
        self.rec_history_phys.append(selected.copy())

    def _build_eval_set(self, n: int = 800, seed: int = 123) -> None:
        rng = np.random.RandomState(seed)
        self.eval_pts_phys = []
        for _ in range(n):
            a = rng.uniform(*self.audio.param_ranges["amplitude"])
            f = rng.uniform(*self.audio.param_ranges["frequency"])
            d = rng.uniform(*self.audio.param_ranges["density"])
            g = rng.uniform(*self.audio.param_ranges["gradient"])
            self.eval_pts_phys.append(np.array([a, f, d, g]))
        self.eval_gt = np.array([self._gt_value(p) for p in self.eval_pts_phys])

    def _update_test_metrics(self, update_rec_best: bool = False) -> None:
        if self.eval_pts_phys is None or self.eval_gt is None or self.gp is None:
            return
        preds = []
        for phys in self.eval_pts_phys:
            norm = self.gp.normalize_parameters(phys)
            mu = self.gp.mean1pt(norm)
            preds.append(float(mu[0] if isinstance(mu, (list, tuple, np.ndarray)) else mu))
        preds = np.asarray(preds)
        gt = self.eval_gt

        if np.std(preds) < 1e-12 or np.std(gt) < 1e-12:
            pear = 0.0
            spear = 0.0
        else:
            pear = float(np.corrcoef(preds, gt)[0, 1])
            sr = spearmanr(preds, gt)
            spear = float(0.0 if sr.correlation is None or np.isnan(sr.correlation) else sr.correlation)
        self.pearson_history.append(pear)
        self.spearman_history.append(spear)

        rng = np.random.RandomState(42)
        N = len(gt)
        idx_i = rng.randint(0, N, size=3500)
        idx_j = rng.randint(0, N, size=3500)
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        gt_diff = gt[idx_i] - gt[idx_j]
        pr_diff = preds[idx_i] - preds[idx_j]
        keep = np.abs(gt_diff) > 1e-12
        acc = float(((gt_diff[keep] > 0) == (pr_diff[keep] > 0)).mean()) if keep.sum() > 0 else 0.5
        self.pairacc_history.append(acc)

        if update_rec_best:
            k = int(np.argmax(preds))
            self.rec_best.parameters = np.asarray(self.eval_pts_phys[k], dtype=float)
            self.rec_best.mean_value = float(preds[k])
            dist = float(np.linalg.norm(self.rec_best.parameters - self.ideal_phys))
            self.rec_distance_history.append(dist)
            self.rec_history_phys.append(self.rec_best.parameters.copy())
            gt_min = float(np.min(gt))
            gt_max = float(np.max(gt))
            gt_range = max(gt_max - gt_min, 1e-12)
            rec_gt_norm = float(np.clip((gt[k] - gt_min) / gt_range, 0.0, 1.0))
            prev = self.best_so_far_history[-1] if self.best_so_far_history else 0.0
            self.best_so_far_history.append(max(prev, rec_gt_norm))

    # ------------------------------------------------------------------ #
    # Ground truth utilities
    # ------------------------------------------------------------------ #
    def _ideal_for_gt_func(self, kind: GroundTruthKind) -> List[float]:
        if kind is GroundTruthKind.GAUSSIAN_OFFSET:
            return [73.3, 44.0, 75.0, 68.0]
        if kind is GroundTruthKind.BIMODAL:
            return [60.0, 60.0, 60.0, 60.0]
        if kind is GroundTruthKind.RIDGE:
            return [60.0, 60.0, 60.0, 60.0]
        return [60.0, 60.0, 60.0, 60.0]

    def _gt_value(self, params_phys: Sequence[float]) -> float:
        kind = self.state.gt_kind
        x = np.asarray(params_phys, dtype=float)
        if kind is GroundTruthKind.GAUSSIAN_OFFSET:
            from scipy.stats import multivariate_normal

            mean = np.array([73.3, 44.0, 75.0, 68.0])
            cov = np.diag([853.3, 640.0, 500.0, 256.0])
            try:
                rv = multivariate_normal(mean=mean, cov=cov)
                return float(rv.pdf(x))
            except Exception:
                d = np.linalg.norm(x - mean)
                return float(np.exp(-d / 28.0))
        if kind is GroundTruthKind.BIMODAL:
            from scipy.stats import multivariate_normal

            mean1 = np.array([60.0, 60.0, 60.0, 60.0])
            mean2 = np.array([86.7, 76.0, 50.0, 72.0])
            cov1 = np.diag([711.0, 563.2, 420.0, 204.8])
            cov2 = np.diag([995.5, 512.0, 360.0, 230.4])
            try:
                rv1 = multivariate_normal(mean=mean1, cov=cov1)
                rv2 = multivariate_normal(mean=mean2, cov=cov2)
                return float(0.6 * rv1.pdf(x) + 0.4 * rv2.pdf(x))
            except Exception:
                d1 = np.linalg.norm(x - mean1)
                d2 = np.linalg.norm(x - mean2)
                return float(0.6 * np.exp(-d1 / 28.0) + 0.4 * np.exp(-d2 / 30.0))
        if kind is GroundTruthKind.RIDGE:
            a, f, d, g = x
            value = (
                np.exp(-((a - 60.0) ** 2) / 480.0)
                * np.exp(-((f - 60.0) ** 2) / 640.0)
                * np.exp(-((d - 60.0) ** 2) / 3000.0)
                * np.exp(-((g - 60.0) ** 2) / 1500.0)
            )
            return float(value)
        from scipy.stats import multivariate_normal

        mean = np.array([60.0, 60.0, 60.0, 60.0])
        cov = np.diag([711.0, 512.0, 400.0, 192.0])
        try:
            rv = multivariate_normal(mean=mean, cov=cov)
            return float(rv.pdf(x))
        except Exception:
            d = np.linalg.norm(x - mean)
            return float(np.exp(-d / 30.0))

    def gt_value(self, params_phys: Sequence[float]) -> float:
        """Public wrapper for ground-truth evaluation."""
        return self._gt_value(params_phys)
