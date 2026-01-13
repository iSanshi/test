"""Launcher for the redesigned Audio Preference Learning UI (automatic test mode)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


CANONICAL_PARAM_ORDER = ("intensity", "texture", "rhythm", "grain")
LEGACY_TO_CANONICAL = {
    "amplitude": "intensity",
    "frequency": "texture",
    "density": "rhythm",
    "gradient": "grain",
}


def _load_param_ranges(raw: Optional[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    if raw is None:
        return None
    path = Path(raw)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = raw
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("param ranges must be a JSON object of key -> [low, high].")
    ranges: Dict[str, Tuple[float, float]] = {}
    for key, bounds in data.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"param range for '{key}' must be a [low, high] pair.")
        low, high = float(bounds[0]), float(bounds[1])
        key = str(key)
        if key in CANONICAL_PARAM_ORDER:
            ranges[key] = (low, high)
            continue
        if key in LEGACY_TO_CANONICAL:
            ranges[LEGACY_TO_CANONICAL[key]] = (low, high)
            continue
        raise ValueError(
            f"Unknown parameter '{key}'. Use {', '.join(CANONICAL_PARAM_ORDER)}."
        )
    return ranges


def _resolve_gt_label(raw: str, ground_truth_kind) -> str:
    aliases = {
        "center": ground_truth_kind.GAUSSIAN_CENTER.value,
        "offset": ground_truth_kind.GAUSSIAN_OFFSET.value,
        "bimodal": ground_truth_kind.BIMODAL.value,
        "ridge": ground_truth_kind.RIDGE.value,
    }
    if raw in aliases:
        return aliases[raw]
    for item in ground_truth_kind:
        if raw == item.value:
            return raw
    raise ValueError(
        "Unknown gt kind. Use center|offset|bimodal|ridge or a full label from GroundTruthKind."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the auto-test UI with custom settings.")
    parser.add_argument("--iters", type=int, default=40, help="Maximum iterations for auto-test.")
    parser.add_argument(
        "--gt",
        type=str,
        default="center",
        #default="center",
        help="Ground-truth kind: center|offset|bimodal|ridge or full label.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for random).")
    parser.add_argument(
        "--ranges",
        type=str,
        default=None,
        help=(
            "JSON string or path for param ranges, "
            "e.g. '{\"intensity\":[20,100],\"texture\":[20,100],...}'."
        ),
    )
    parser.add_argument("--plot-res", type=int, default=31, help="Grid resolution for map plots.")
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="Only redraw map every N iterations.",
    )
    parser.add_argument(
        "--plot-min-s",
        type=float,
        default=0.0,
        help="Minimum seconds between map redraws.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from preference_learning.interface.session import GroundTruthKind
    from preference_learning.interface.ui_study import test_main

    try:
        param_ranges = _load_param_ranges(args.ranges)
        gt_label = _resolve_gt_label(args.gt, GroundTruthKind)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    return test_main(
        max_iterations=args.iters,
        gt_label=gt_label,
        seed=args.seed,
        param_ranges=param_ranges,
        plot_resolution=args.plot_res,
        plot_update_every=args.plot_every,
        plot_min_interval_s=args.plot_min_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
