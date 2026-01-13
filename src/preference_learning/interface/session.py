"""
Session/business logic for the audio preference learning application.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import AudioGenerator
from ..gp import AudioPreferenceGaussianProcess
from ..evaluation import (
    correlation_metrics,
    find_strong_ground_truth,
    pairwise_accuracy,
    posterior_mean,
    posterior_uncertainty,
    recursive_sanitize,
    test_metrics,
)


LEVEL_TO_WEIGHT = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0}
DIFF_THRESHOLDS = [0.01, 0.05, 0.10, 0.20]
DEFAULT_SESSION_SEED = None
DEFAULT_VALIDATION_ROUNDS = 5
DEFAULT_GT_RANDOM_SAMPLES = 10000
DEFAULT_GT_TOP_K = 20
DEFAULT_VALIDATION_TOP_K = 20
DEFAULT_VALIDATION_RANDOM_TRIALS = 200
PARAMETER_NAME_MAP = {
    "intensity": "amplitude",
    "texture": "frequency",
    "rhythm": "density",
    "grain": "gradient",
}


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


class SessionPhase(Enum):
    TRAINING = "Training"
    VALIDATION = "Validation"


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
class UserQueryRecord:
    iteration: int
    normalized: Tuple[np.ndarray, np.ndarray]
    physical: Tuple[np.ndarray, np.ndarray]
    choice: str
    level: int
    info_gain: float
    query_distance: float


@dataclass
class Recommendation:
    parameters: Optional[np.ndarray] = None
    mean_value: Optional[float] = None
    method: Optional[str] = None


@dataclass
class ValidationRecord:
    round_index: int
    choice: str
    level: int
    normalized: Optional[Tuple[np.ndarray, np.ndarray]] = None
    physical: Optional[Tuple[np.ndarray, np.ndarray]] = None
    query_distance: Optional[float] = None
    pair_type: Optional[str] = None
    predicted_margin: Optional[float] = None
    model_winner: Optional[str] = None
    is_aligned: Optional[bool] = None


@dataclass
class SessionState:
    mode: SessionMode = SessionMode.USER
    max_iterations: int = 40
    current_iteration: int = 0
    running: bool = False
    gt_kind: GroundTruthKind = GroundTruthKind.GAUSSIAN_CENTER
    seed: Optional[int] = None
    phase: SessionPhase = SessionPhase.TRAINING
    validation_rounds: int = DEFAULT_VALIDATION_ROUNDS
    validation_index: int = 0


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
        self.posterior_best_mean_history: List[float] = []
        self.posterior_rec_mean_history: List[float] = []
        self.gt_regret_history: List[float] = []
        self.gt_spearman_history: List[float] = []
        self.user_query_history: List[UserQueryRecord] = []
        self.test_query_history: List[TestQueryRecord] = []
        self.validation_records: List[ValidationRecord] = []
        self.validation_recommended: Optional[np.ndarray] = None
        self.validation_competitor: Optional[np.ndarray] = None
        self.validation_config: Optional[Dict[str, object]] = None
        self.validation_pairs: List[Dict[str, object]] = []

        self.gt_best_val: Optional[float] = None
        self.gt_best_params: Optional[np.ndarray] = None
        self.eval_set_best_val: Optional[float] = None
        self.gt_search_config: Optional[Dict[str, object]] = None

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
        self.posterior_best_mean_history.clear()
        self.posterior_rec_mean_history.clear()
        self.gt_regret_history.clear()
        self.gt_spearman_history.clear()
        self.user_query_history.clear()
        self.test_query_history.clear()
        self.validation_records.clear()
        self.validation_recommended = None
        self.validation_competitor = None
        self.validation_config = None
        self.validation_pairs.clear()
        self.gt_best_val = None
        self.gt_best_params = None
        self.eval_set_best_val = None
        self.gt_search_config = None
        self.eval_pts_phys = None
        self.eval_gt = None

        self.state.current_iteration = 0
        self.state.running = False
        self.state.seed = None
        self.state.phase = SessionPhase.TRAINING
        self.state.validation_rounds = DEFAULT_VALIDATION_ROUNDS
        self.state.validation_index = 0

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            chosen = DEFAULT_SESSION_SEED
            if chosen is None:
                chosen = int(secrets.randbits(32))
        else:
            chosen = int(seed)
        self.state.seed = chosen
        np.random.seed(chosen)

    def start(
        self,
        mode: SessionMode,
        max_iterations: int,
        gt_label: str,
        seed: Optional[int] = None,
        validation_rounds: int = DEFAULT_VALIDATION_ROUNDS,
    ) -> None:
        self.reset()
        self._set_seed(seed)
        self.state.mode = mode
        self.state.max_iterations = max(1, max_iterations)
        self.state.gt_kind = GroundTruthKind.from_label(gt_label)
        self.state.phase = SessionPhase.TRAINING
        self.state.validation_rounds = max(0, int(validation_rounds))
        self.state.validation_index = 0

        initial_point = [0.5, 0.5, 0.5, 0.5]
        parameter_bounds = {
            key: (float(bounds[0]), float(bounds[1]))
            for key, bounds in self.audio.param_ranges.items()
        }
        self.gp = AudioPreferenceGaussianProcess(
            initial_point=initial_point,
            theta=2.5,
            noise_level=0.1,
            parameter_bounds=parameter_bounds,
        )

        center = np.array(initial_point)
        self.pref_dict = {tuple(center): LEVEL_TO_WEIGHT[3]}
        self.gp.update_parameters([center, center], 0, 3, self.pref_dict)

        self.state.running = True
        self.ideal_phys = np.array(self._ideal_for_gt_func(self.state.gt_kind), dtype=float)

        if self.state.mode is SessionMode.TEST:
            self._build_eval_set(n=800)
            if self.eval_gt is not None and len(self.eval_gt) > 0:
                self.eval_set_best_val = float(np.max(self.eval_gt))
            bounds = self._physical_bounds()
            gt_best_val, gt_best_params = find_strong_ground_truth(
                self._gt_value,
                bounds,
                n_random=DEFAULT_GT_RANDOM_SAMPLES,
                top_k=DEFAULT_GT_TOP_K,
            )
            self.gt_best_val = float(gt_best_val)
            self.gt_best_params = np.asarray(gt_best_params, dtype=float)
            self.gt_search_config = {
                "n_random": int(DEFAULT_GT_RANDOM_SAMPLES),
                "top_k": int(DEFAULT_GT_TOP_K),
                "method": "lbfgsb",
            }

    def stop(self) -> None:
        self.state.running = False
        self._stop_event.set()

    def training_complete(self) -> bool:
        return self.state.current_iteration >= self.state.max_iterations

    def validation_complete(self) -> bool:
        return self.state.validation_index >= self.state.validation_rounds

    def is_complete(self) -> bool:
        if self.state.mode is SessionMode.USER:
            if not self.training_complete():
                return False
            if self.state.validation_rounds <= 0:
                return True
            return self.validation_complete()
        return self.training_complete()

    def start_validation(self, rounds: Optional[int] = None) -> None:
        if self.state.mode is not SessionMode.USER or self.gp is None:
            return
        requested_rounds = int(rounds) if rounds is not None else int(self.state.validation_rounds)
        if requested_rounds <= 0:
            return
        self.state.phase = SessionPhase.VALIDATION
        self.state.validation_index = 0
        self.state.validation_rounds = max(1, int(requested_rounds))
        self.validation_records.clear()
        self.validation_pairs.clear()

        if self.rec_best.parameters is None or self.rec_best.mean_value is None:
            self.rec_best = self._compute_recommendation()
        if self.rec_best.parameters is None:
            return
        self.validation_recommended = np.asarray(self.rec_best.parameters, dtype=float)

        bounds = self._physical_bounds()
        bounds_arr = np.asarray(bounds, dtype=float)
        diag = float(np.linalg.norm(bounds_arr[:, 1] - bounds_arr[:, 0]))
        threshold = 0.05 * diag
        relaxed_threshold = 0.03 * diag
        max_attempts = int(DEFAULT_VALIDATION_RANDOM_TRIALS)

        def sample_point() -> np.ndarray:
            return np.array([np.random.uniform(low, high) for low, high in bounds], dtype=float)

        def min_dist(cand: np.ndarray, pts: Sequence[np.ndarray]) -> float:
            return min(float(np.linalg.norm(cand - p)) for p in pts)

        def far_corner(rec: np.ndarray) -> np.ndarray:
            lows = bounds_arr[:, 0]
            highs = bounds_arr[:, 1]
            choice = np.where((rec - lows) >= (highs - rec), lows, highs)
            return np.asarray(choice, dtype=float)

        points: List[np.ndarray] = [self.validation_recommended.copy()]
        primary_attempts = 0
        while len(points) < 7 and primary_attempts < max_attempts:
            cand = sample_point()
            if min_dist(cand, points) >= threshold:
                points.append(cand)
            primary_attempts += 1

        relaxed_attempts = 0
        if len(points) < 7:
            while len(points) < 7 and relaxed_attempts < max_attempts:
                cand = sample_point()
                if min_dist(cand, points) >= relaxed_threshold:
                    points.append(cand)
                relaxed_attempts += 1

        fallback_attempts = 0
        fallback_used = False
        if len(points) < 7:
            fallback_used = True
            while len(points) < 7:
                best_cand = None
                best_min = -1.0
                for _ in range(max_attempts):
                    cand = sample_point()
                    if float(np.linalg.norm(cand - points[0])) < threshold:
                        continue
                    dist_val = min_dist(cand, points)
                    if dist_val > best_min:
                        best_min = dist_val
                        best_cand = cand
                if best_cand is None:
                    best_cand = far_corner(points[0])
                points.append(best_cand)
                fallback_attempts += max_attempts

        means = np.asarray(posterior_mean(self.gp, points), dtype=float)
        sorted_idx = np.argsort(-means)
        points_sorted = [points[i] for i in sorted_idx]
        means_sorted = [float(means[i]) for i in sorted_idx]

        def make_pair(
            a: np.ndarray,
            b: np.ndarray,
            mean_a: float,
            mean_b: float,
            pair_type: str,
        ) -> Dict[str, object]:
            model_winner = "A" if mean_a > mean_b else "B"
            margin = abs(mean_a - mean_b)
            return {
                "A": np.asarray(a, dtype=float),
                "B": np.asarray(b, dtype=float),
                "pair_type": pair_type,
                "mean_a": float(mean_a),
                "mean_b": float(mean_b),
                "predicted_margin": float(margin),
                "model_winner": model_winner,
            }

        best = points_sorted[0]
        worst = points_sorted[-1]
        mid_idx = len(points_sorted) // 2
        mid = points_sorted[mid_idx]

        round1 = make_pair(best, worst, means_sorted[0], means_sorted[-1], "anchor_easy")
        round2 = make_pair(best, mid, means_sorted[0], means_sorted[mid_idx], "anchor_medium")

        hard_idx = None
        hard_margin = None
        for i in range(len(points_sorted) - 1):
            if {i, i + 1} == {0, mid_idx}:
                continue
            margin = abs(means_sorted[i] - means_sorted[i + 1])
            if hard_margin is None or margin < hard_margin:
                hard_margin = margin
                hard_idx = i
        if hard_idx is None:
            hard_idx = 0
        round3 = make_pair(
            points_sorted[hard_idx],
            points_sorted[hard_idx + 1],
            means_sorted[hard_idx],
            means_sorted[hard_idx + 1],
            "hard_local",
        )

        max_mean = max(means_sorted)
        min_mean = min(means_sorted)
        margin_threshold = max(0.3, 0.2 * (max_mean - min_mean))
        margin_threshold_fallback = 0.15

        def select_global_tradeoff(margin_th: float) -> Optional[Tuple[int, int]]:
            best_pair = None
            best_margin = -1.0
            best_dist = -1.0
            for i in range(1, len(points_sorted) - 1):
                for j in range(i + 1, len(points_sorted)):
                    dist_val = float(np.linalg.norm(points_sorted[i] - points_sorted[j]))
                    if dist_val < threshold:
                        continue
                    margin_val = abs(means_sorted[i] - means_sorted[j])
                    if margin_val < margin_th:
                        continue
                    if margin_val > best_margin or (margin_val == best_margin and dist_val > best_dist):
                        best_pair = (i, j)
                        best_margin = margin_val
                        best_dist = dist_val
            return best_pair

        global_pair = select_global_tradeoff(margin_threshold)
        global_strategy = "threshold"
        if global_pair is None:
            global_pair = select_global_tradeoff(margin_threshold_fallback)
            global_strategy = "margin_fallback"

        if global_pair is not None:
            i, j = global_pair
            round4 = make_pair(
                points_sorted[i],
                points_sorted[j],
                means_sorted[i],
                means_sorted[j],
                "global_tradeoff",
            )
        else:
            fallback_pair = (1, len(points_sorted) - 2)
            dist_val = float(
                np.linalg.norm(points_sorted[fallback_pair[0]] - points_sorted[fallback_pair[1]])
            )
            if dist_val >= threshold:
                round4 = make_pair(
                    points_sorted[fallback_pair[0]],
                    points_sorted[fallback_pair[1]],
                    means_sorted[fallback_pair[0]],
                    means_sorted[fallback_pair[1]],
                    "global_tradeoff",
                )
                global_strategy = "fallback_pair"
            else:
                a = sample_point()
                b = sample_point()
                best_dist = float(np.linalg.norm(a - b))
                for _ in range(max_attempts):
                    cand_a = sample_point()
                    cand_b = sample_point()
                    dist_val = float(np.linalg.norm(cand_a - cand_b))
                    if dist_val >= threshold:
                        a, b = cand_a, cand_b
                        best_dist = dist_val
                        break
                    if dist_val > best_dist:
                        a, b = cand_a, cand_b
                        best_dist = dist_val
                mean_a, mean_b = posterior_mean(self.gp, [a, b])
                round4 = make_pair(a, b, float(mean_a), float(mean_b), "global_tradeoff")
                global_strategy = "random_far"

        round5 = make_pair(
            round2["B"],
            round2["A"],
            round2["mean_b"],
            round2["mean_a"],
            "consistency_check",
        )

        self.validation_pairs = [round1, round2, round3, round4, round5]
        if self.state.validation_rounds < len(self.validation_pairs):
            self.validation_pairs = self.validation_pairs[: self.state.validation_rounds]
        elif self.state.validation_rounds > len(self.validation_pairs):
            self.state.validation_rounds = len(self.validation_pairs)
        self.validation_competitor = np.asarray(worst, dtype=float)
        self.validation_config = {
            "strategy": "smcc",
            "requested_rounds": int(requested_rounds),
            "rounds": int(self.state.validation_rounds),
            "set_size": int(len(points)),
            "diag": float(diag),
            "threshold": float(threshold),
            "relaxed_threshold": float(relaxed_threshold),
            "max_attempts": int(max_attempts),
            "sampling_attempts": {
                "primary": int(primary_attempts),
                "relaxed": int(relaxed_attempts),
                "fallback": int(fallback_attempts),
                "fallback_used": bool(fallback_used),
            },
            "margin_threshold": float(margin_threshold),
            "margin_threshold_fallback": float(margin_threshold_fallback),
            "global_tradeoff_strategy": global_strategy,
            "pair_types": [pair["pair_type"] for pair in self.validation_pairs],
        }

    # ------------------------------------------------------------------ #
    # User mode helpers
    # ------------------------------------------------------------------ #
    def generate_user_query(self) -> Optional[AudioCandidate]:
        if (
            not self.state.running
            or self.state.mode is not SessionMode.USER
            or self.state.phase is not SessionPhase.TRAINING
            or self.gp is None
        ):
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
        self.query_distance_history.append(dist)
        return candidate

    def generate_validation_query(self) -> Optional[AudioCandidate]:
        if (
            not self.state.running
            or self.state.mode is not SessionMode.USER
            or self.state.phase is not SessionPhase.VALIDATION
            or self.gp is None
        ):
            return None
        if self.validation_pairs:
            if self.state.validation_index >= len(self.validation_pairs):
                return None
            pair = self.validation_pairs[self.state.validation_index]
            p1_phys = np.asarray(pair["A"], dtype=float)
            p2_phys = np.asarray(pair["B"], dtype=float)
        else:
            if self.validation_recommended is None or self.validation_competitor is None:
                return None
            p1_phys = np.asarray(self.validation_recommended, dtype=float)
            p2_phys = np.asarray(self.validation_competitor, dtype=float)
        p1_norm = self.gp.normalize_parameters(p1_phys)
        p2_norm = self.gp.normalize_parameters(p2_phys)
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
            info_gain=float("nan"),
            query_distance=dist,
            audio_data=audio_data,
        )
        self.current_candidates = candidate
        return candidate

    def record_user_choice(self, choice_label: str, level: int) -> None:
        if not self.current_candidates or self.gp is None:
            return

        level = int(np.clip(level, 1, 5))
        y = 1 if choice_label == "A" else -1
        p1_norm, p2_norm = self.current_candidates.normalized
        p1_phys, p2_phys = self.current_candidates.physical
        p1_norm = np.asarray(p1_norm, dtype=float)
        p2_norm = np.asarray(p2_norm, dtype=float)
        p1_phys = np.asarray(p1_phys, dtype=float)
        p2_phys = np.asarray(p2_phys, dtype=float)

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

        self.info_gain_history.append(float(self.current_candidates.info_gain))
        self.query_distance_history.append(self.current_candidates.query_distance)

        self.state.current_iteration += 1
        self.user_query_history.append(
            UserQueryRecord(
                iteration=self.state.current_iteration,
                normalized=(p1_norm.copy(), p2_norm.copy()),
                physical=(p1_phys.copy(), p2_phys.copy()),
                choice=choice_label,
                level=level,
                info_gain=float(self.current_candidates.info_gain),
                query_distance=float(self.current_candidates.query_distance),
            )
        )
        self._update_recommendation()

    def record_validation_choice(self, choice_label: str, level: int) -> None:
        if self.state.phase is not SessionPhase.VALIDATION:
            return
        level = int(np.clip(level, 1, 5))
        pair_meta = None
        if 0 <= self.state.validation_index < len(self.validation_pairs):
            pair_meta = self.validation_pairs[self.state.validation_index]
        pair_type = pair_meta.get("pair_type") if pair_meta else None
        predicted_margin = pair_meta.get("predicted_margin") if pair_meta else None
        model_winner = pair_meta.get("model_winner") if pair_meta else None
        is_aligned = None
        if model_winner in ("A", "B"):
            is_aligned = choice_label == model_winner
        p1_norm = p2_norm = None
        p1_phys = p2_phys = None
        query_distance = None
        if self.current_candidates is not None:
            cand_norm = self.current_candidates.normalized
            cand_phys = self.current_candidates.physical
            p1_norm = np.asarray(cand_norm[0], dtype=float)
            p2_norm = np.asarray(cand_norm[1], dtype=float)
            p1_phys = np.asarray(cand_phys[0], dtype=float)
            p2_phys = np.asarray(cand_phys[1], dtype=float)
            query_distance = float(self.current_candidates.query_distance)
        elif pair_meta is not None:
            if "A" in pair_meta and "B" in pair_meta:
                p1_phys = np.asarray(pair_meta["A"], dtype=float)
                p2_phys = np.asarray(pair_meta["B"], dtype=float)
                if self.gp is not None:
                    p1_norm = self.gp.normalize_parameters(p1_phys)
                    p2_norm = self.gp.normalize_parameters(p2_phys)
                    query_distance = float(np.linalg.norm(p1_norm - p2_norm))

        record = ValidationRecord(
            round_index=self.state.validation_index + 1,
            choice=choice_label,
            level=level,
            normalized=(p1_norm.copy(), p2_norm.copy()) if p1_norm is not None else None,
            physical=(p1_phys.copy(), p2_phys.copy()) if p1_phys is not None else None,
            query_distance=query_distance,
            pair_type=pair_type,
            predicted_margin=predicted_margin,
            model_winner=model_winner,
            is_aligned=is_aligned,
        )
        self.validation_records.append(record)
        self.state.validation_index += 1

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
                    self._build_eval_set(n=800)
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
    def _compute_recommendation(self) -> Recommendation:
        if self.gp is None:
            return Recommendation()
        params, mean_val, method = self.gp.find_recommendation()
        return Recommendation(parameters=np.asarray(params, dtype=float), mean_value=float(mean_val), method=method)

    def _append_posterior_best_mean(self, current_mean: float) -> None:
        if self.posterior_best_mean_history:
            prev = self.posterior_best_mean_history[-1]
            self.posterior_best_mean_history.append(max(prev, current_mean))
        else:
            self.posterior_best_mean_history.append(current_mean)

    def _update_recommendation(self) -> None:
        if self.gp is None:
            return
        self.rec_best = self._compute_recommendation()
        if self.rec_best.parameters is None or self.rec_best.mean_value is None:
            return
        self.rec_history_phys.append(self.rec_best.parameters.copy())
        self.posterior_rec_mean_history.append(float(self.rec_best.mean_value))
        self._append_posterior_best_mean(float(self.rec_best.mean_value))

    def _build_eval_set(self, n: int = 800) -> None:
        self.eval_pts_phys = []
        for _ in range(n):
            a = np.random.uniform(*self.audio.param_ranges["amplitude"])
            f = np.random.uniform(*self.audio.param_ranges["frequency"])
            d = np.random.uniform(*self.audio.param_ranges["density"])
            g = np.random.uniform(*self.audio.param_ranges["gradient"])
            self.eval_pts_phys.append(np.array([a, f, d, g]))
        self.eval_gt = np.array([self._gt_value(p) for p in self.eval_pts_phys])

    def _update_test_metrics(self, update_rec_best: bool = False) -> None:
        if self.eval_pts_phys is None or self.eval_gt is None or self.gp is None:
            return
        preds = posterior_mean(self.gp, self.eval_pts_phys)
        gt = np.asarray(self.eval_gt, dtype=float)

        corr = correlation_metrics(preds, gt)
        self.pearson_history.append(corr["pearson"])
        self.spearman_history.append(corr["spearman"])
        self.gt_spearman_history.append(corr["spearman"])
        self.pairacc_history.append(pairwise_accuracy(preds, gt))

        if update_rec_best:
            self.rec_best = self._compute_recommendation()
            if self.rec_best.parameters is None or self.rec_best.mean_value is None:
                return
            self.rec_history_phys.append(self.rec_best.parameters.copy())
            self.posterior_rec_mean_history.append(float(self.rec_best.mean_value))
            self._append_posterior_best_mean(float(self.rec_best.mean_value))
            dist = float(np.linalg.norm(self.rec_best.parameters - self.ideal_phys))
            self.rec_distance_history.append(dist)
            gt_min = float(np.min(gt))
            gt_max = float(np.max(gt))
            gt_range = max(gt_max - gt_min, 1e-12)
            rec_gt = self._gt_value(self.rec_best.parameters)
            rec_gt_norm = float(np.clip((rec_gt - gt_min) / gt_range, 0.0, 1.0))
            prev = self.best_so_far_history[-1] if self.best_so_far_history else 0.0
            self.best_so_far_history.append(max(prev, rec_gt_norm))

            gt_best_val = self.gt_best_val if self.gt_best_val is not None else self.eval_set_best_val
            if gt_best_val is None:
                regret = 0.0
            else:
                gt_rec_val = float(self._gt_value(self.rec_best.parameters))
                regret = float(max(0.0, (gt_best_val - gt_rec_val) / (abs(gt_best_val) + 1e-12)))
            self.gt_regret_history.append(regret)

    def _physical_bounds(self) -> List[Tuple[float, float]]:
        ranges = self.audio.param_ranges
        return [
            ranges["amplitude"],
            ranges["frequency"],
            ranges["density"],
            ranges["gradient"],
        ]

    def _bounds_dict(self) -> Dict[str, List[float]]:
        ranges = self.audio.param_ranges
        bounds_dict = {}
        for canonical, legacy in PARAMETER_NAME_MAP.items():
            bounds = ranges.get(legacy)
            if bounds is None:
                continue
            bounds_dict[canonical] = [float(bounds[0]), float(bounds[1])]
        return bounds_dict

    def _build_validation_summary(self) -> Dict[str, object]:
        rounds = len(self.validation_records)
        wins = sum(1 for rec in self.validation_records if getattr(rec, "choice", "") == "A")
        win_rate = wins / rounds if rounds else 0.0

        records = []
        aligned_count = 0
        margin_sum = 0.0
        weighted_sum = 0.0
        round2_choice = None
        round5_choice = None

        for rec in self.validation_records:
            predicted_margin = float(rec.predicted_margin) if rec.predicted_margin is not None else None
            records.append(
                {
                    "round": int(rec.round_index),
                    "choice": rec.choice,
                    "level": int(rec.level),
                    "pair_type": rec.pair_type,
                    "predicted_margin": predicted_margin,
                    "model_winner": rec.model_winner,
                    "is_aligned": rec.is_aligned,
                }
            )
            if rec.is_aligned:
                aligned_count += 1
            if predicted_margin is not None:
                margin_sum += predicted_margin
                if rec.is_aligned:
                    weighted_sum += predicted_margin
            if rec.pair_type == "anchor_medium":
                round2_choice = rec.choice
            if rec.pair_type == "consistency_check":
                round5_choice = rec.choice

        planned_rounds = max(1, int(self.state.validation_rounds))
        agreement_rate = aligned_count / float(planned_rounds) if planned_rounds else 0.0
        weighted_agreement = weighted_sum / margin_sum if margin_sum > 0 else None

        consistency_pass = None
        if round2_choice in ("A", "B") and round5_choice in ("A", "B"):
            consistency_pass = (round2_choice == "A" and round5_choice == "B") or (
                round2_choice == "B" and round5_choice == "A"
            )

        return {
            "rounds": rounds,
            "win_rate": float(win_rate),
            "records": records,
            "recommended_params": (
                list(map(float, np.asarray(self.validation_recommended, dtype=float).tolist()))
                if self.validation_recommended is not None
                else None
            ),
            "competitor_params": (
                list(map(float, np.asarray(self.validation_competitor, dtype=float).tolist()))
                if self.validation_competitor is not None
                else None
            ),
            "agreement_rate": float(agreement_rate),
            "weighted_agreement": float(weighted_agreement) if weighted_agreement is not None else None,
            "consistency_pass": consistency_pass,
        }

    def build_snapshot(self, status: str = "complete") -> Dict[str, object]:
        recommended = self.rec_best
        if (recommended.parameters is None or recommended.mean_value is None) and self.gp is not None:
            recommended = self._compute_recommendation()

        rec_params = (
            list(map(float, np.asarray(recommended.parameters, dtype=float).tolist()))
            if recommended.parameters is not None
            else None
        )
        rec_score = float(recommended.mean_value) if recommended.mean_value is not None else None

        bounds_list = self._physical_bounds()
        bounds_dict = self._bounds_dict()
        if self.gp is not None:
            uncertainty = posterior_uncertainty(self.gp, bounds_list, n_samples=500)
        else:
            uncertainty = {"avg_pred_var": None, "max_pred_var": None}

        final_summary: Dict[str, object] = {
            "recommended_params": rec_params,
            "recommended_score": rec_score,
            "method": recommended.method,
            "bounds": bounds_dict,
            "posterior_uncertainty": uncertainty,
        }

        if self.state.mode is SessionMode.USER:
            final_summary["validation"] = self._build_validation_summary()
            final_summary["validation_config"] = self.validation_config

        if (
            self.state.mode is SessionMode.TEST
            and self.gp is not None
            and self.eval_pts_phys is not None
            and self.eval_gt is not None
            and recommended.parameters is not None
        ):
            gt_rec_val = float(self._gt_value(recommended.parameters))
            gt_best_params = (
                np.asarray(self.gt_best_params, dtype=float) if self.gt_best_params is not None else None
            )
            gt_best_val = float(self.gt_best_val) if self.gt_best_val is not None else None
            eval_set_best_val = float(self.eval_set_best_val) if self.eval_set_best_val is not None else None
            final_summary.update(
                {
                    "gt_best_val": gt_best_val,
                    "gt_best_params": (
                        list(map(float, gt_best_params.tolist())) if gt_best_params is not None else None
                    ),
                    "gt_rec_val": gt_rec_val,
                    "eval_set_best_val": eval_set_best_val,
                    "gt_search_config": self.gt_search_config,
                }
            )
            true_optimum_params = gt_best_params if gt_best_params is not None else self.ideal_phys
            final_summary["test_metrics"] = test_metrics(
                self.gp,
                self.eval_pts_phys,
                self.eval_gt,
                recommended.parameters,
                true_optimum_params=true_optimum_params,
                gt_eval_fn=self._gt_value,
            )

        def to_float_list(arr: Optional[np.ndarray]) -> Optional[List[float]]:
            if arr is None:
                return None
            return list(map(float, np.asarray(arr, dtype=float).tolist()))

        training_queries = []
        for rec in self.user_query_history:
            p1_norm, p2_norm = rec.normalized
            p1_phys, p2_phys = rec.physical
            training_queries.append(
                {
                    "iteration": int(rec.iteration),
                    "A": to_float_list(p1_phys),
                    "B": to_float_list(p2_phys),
                    "A_norm": to_float_list(p1_norm),
                    "B_norm": to_float_list(p2_norm),
                    "choice": rec.choice,
                    "level": int(rec.level),
                    "info_gain": float(rec.info_gain),
                    "query_distance": float(rec.query_distance),
                }
            )

        validation_queries = []
        for idx, rec in enumerate(self.validation_records):
            p1_norm, p2_norm = None, None
            p1_phys, p2_phys = None, None
            if rec.normalized is not None:
                p1_norm, p2_norm = rec.normalized
            if rec.physical is not None:
                p1_phys, p2_phys = rec.physical
            if (p1_phys is None or p2_phys is None) and idx < len(self.validation_pairs):
                pair_meta = self.validation_pairs[idx]
                if "A" in pair_meta and "B" in pair_meta:
                    p1_phys = np.asarray(pair_meta["A"], dtype=float)
                    p2_phys = np.asarray(pair_meta["B"], dtype=float)
            if (p1_norm is None or p2_norm is None) and self.gp is not None and p1_phys is not None:
                p1_norm = self.gp.normalize_parameters(p1_phys)
                p2_norm = self.gp.normalize_parameters(p2_phys)
            query_distance = rec.query_distance
            if query_distance is None and p1_norm is not None and p2_norm is not None:
                query_distance = float(np.linalg.norm(np.asarray(p1_norm) - np.asarray(p2_norm)))
            validation_queries.append(
                {
                    "round": int(rec.round_index),
                    "A": to_float_list(p1_phys),
                    "B": to_float_list(p2_phys),
                    "A_norm": to_float_list(p1_norm),
                    "B_norm": to_float_list(p2_norm),
                    "choice": rec.choice,
                    "level": int(rec.level),
                    "pair_type": rec.pair_type,
                    "predicted_margin": (
                        float(rec.predicted_margin) if rec.predicted_margin is not None else None
                    ),
                    "model_winner": rec.model_winner,
                    "is_aligned": rec.is_aligned,
                    "query_distance": query_distance,
                }
            )

        test_queries = []
        for rec in self.test_query_history:
            p1_phys, p2_phys = rec.physical
            p1_norm = p2_norm = None
            if self.gp is not None:
                p1_norm = self.gp.normalize_parameters(p1_phys)
                p2_norm = self.gp.normalize_parameters(p2_phys)
            test_queries.append(
                {
                    "iteration": int(rec.iteration),
                    "A": to_float_list(p1_phys),
                    "B": to_float_list(p2_phys),
                    "A_norm": to_float_list(p1_norm),
                    "B_norm": to_float_list(p2_norm),
                    "choice": rec.choice,
                    "level": int(rec.level),
                    "info_gain": float(rec.info_gain),
                    "query_distance": float(rec.query_distance),
                }
            )

        query_pairs = {
            "training": training_queries,
            "validation": validation_queries,
            "test": test_queries,
        }

        metrics = {
            "info_gain": list(map(float, self.info_gain_history)),
            "posterior_rec_mean": list(map(float, self.posterior_rec_mean_history)),
            "posterior_best_mean": list(map(float, self.posterior_best_mean_history)),
            "gt_regret_history": list(map(float, self.gt_regret_history)),
            "gt_spearman_history": list(map(float, self.gt_spearman_history)),
        }

        param_ranges: Dict[str, List[float]] = {}
        for canonical, legacy in PARAMETER_NAME_MAP.items():
            bounds = self.audio.param_ranges.get(legacy)
            if bounds is None:
                bounds = self.audio.param_ranges.get(canonical)
            if bounds is None:
                continue
            param_ranges[canonical] = [float(bounds[0]), float(bounds[1])]
        for key, bounds in self.audio.param_ranges.items():
            if key in PARAMETER_NAME_MAP.values():
                continue
            if key in param_ranges:
                continue
            param_ranges[key] = [float(bounds[0]), float(bounds[1])]

        metadata = {
            "mode": self.state.mode.value,
            "n_queries_planned": int(self.state.max_iterations),
            "n_queries_completed": int(self.state.current_iteration),
            "status": status,
            "seed": int(self.state.seed) if self.state.seed is not None else None,
            "gt_kind": self.state.gt_kind.value,
            "phase": self.state.phase.value,
            "validation_rounds": int(self.state.validation_rounds),
            "param_ranges": param_ranges,
        }

        snapshot: Dict[str, object] = {
            "mode": self.state.mode.value,
            "max_iterations": self.state.max_iterations,
            "completed_iterations": self.state.current_iteration,
            "preferences": self.preference_history,
            "uncertainties": self.uncertainty_history,
            "info_gain": self.info_gain_history,
            "query_distance": self.query_distance_history,
            "parameters": [list(map(float, np.asarray(p).tolist())) for p in self.parameter_history_phys],
            "recommendation_history": [
                list(map(float, np.asarray(p).tolist())) for p in self.rec_history_phys
            ],
            "final_recommendation": rec_params,
            "final_summary": final_summary,
            "query_pairs": query_pairs,
            "metrics": metrics,
            "metadata": metadata,
        }
        snapshot = recursive_sanitize(snapshot)
        json.dumps(snapshot, allow_nan=False)
        return snapshot

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
