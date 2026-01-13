# Haptic Preference Learning (HPL)

[中文版](README_zh-CN.md)

This repo provides a **preference-based haptic personalization** framework that learns a user's latent utility from **binary A/B choices**. We use a **Gaussian Process (GP) preference model** to capture smoothness and uncertainty over the stimulus space, and an **active query policy** that maximizes **expected information gain** to pick the next comparison. Users can report **response uncertainty**, which is used as per-comparison weights to down-weight ambiguous judgments. By emphasizing **relative** (not absolute) evaluations, the system reduces rating fatigue and drift and avoids forcing tactile sensations onto a numeric scale.

**Highlights**
- GP preference learning over haptic stimuli (uncertainty-aware, smoothness prior)
- Information-gain active querying for sample-efficient searches
- Per-comparison **uncertainty weighting** to handle ambiguous answers
- End-to-end UI for user study, auto-test, and favorite-signal capture

![UI Demo](image1.png)
![UI Demo](image2.png)
## Quick Start
1. Clone the repo:
   ```bash
   git clone https://github.com/iSanshi/haptic-preference-learning.git
   cd haptic-preference-learning
   ```
2. (Optional) create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch:
   ```bash
   python run_study.py         # user study + favorite signal capture
   python run_user_study_ui.py # user study only
   python run_auto_test_ui.py  # automated test workflow
   python xbox_control.py      # standalone favorite capture
   ```
5. Press **Start Session** in the UI. In user mode, play A/B (X/Y), choose A/B, and rate uncertainty (1-5). In auto-test mode, the system simulates preferences via a ground-truth function.
6. When a session completes, results are exported to `data/YYYYMMDD_<index>/session.json` and `log.txt`.

## Outputs
- `data/YYYYMMDD_<index>/session.json` + `log.txt`: preference history, GP metrics, and final summary.
- `data/YYYYMMDD_<index>/favorite_signal.json`: saved by `run_study.py` after you record the favorite signal.
- `data/bestparam/###.json`: saved when running `xbox_control.py` standalone.

## Configuration
- User-study iterations: `DEFAULT_MAX_ITERS` in `src/preference_learning/interface/ui_study.py`.
- Auto-test iterations: `--iters` in `run_auto_test_ui.py` (default 40).
- Ground-truth model: `--gt` (`center|offset|bimodal|ridge`).
- Parameter ranges: `--ranges` accepts JSON with keys `intensity|texture|rhythm|grain` (legacy keys are accepted too).

## Requirements
- Python 3.8+ with Tkinter.
- Audio output via PortAudio (`sounddevice`).
- Optional Xbox controller via `pygame`.

## Session Output (session.json)
The export keeps legacy fields and adds structured summaries:
- `final_summary`: GP posterior-mean recommendation, search method, bounds, posterior uncertainty, and (mode-dependent) validation/test metrics.
- `metrics`: per-iteration arrays such as `info_gain` (aligned to preferences) and `posterior_best_mean`.
- `metadata`: session mode, planned/completed queries, and completion status.
- Additional test/validation fields: `gt_best_val`, `gt_best_params`, `gt_rec_val`, `eval_set_best_val`, `gt_search_config`, `validation_config`, `gt_regret_history`, `gt_spearman_history`.

Example snippet:
```json
{
  "final_summary": {
    "recommended_params": [61.2, 58.7, 64.0, 55.9],
    "recommended_score": 0.84,
    "method": "lbfgsb",
    "bounds": {"intensity": [20.0, 100.0], "texture": [20.0, 100.0], "rhythm": [20.0, 100.0], "grain": [20.0, 100.0]},
    "posterior_uncertainty": {"avg_pred_var": 0.12, "max_pred_var": 0.41},
    "validation": {"rounds": 3, "win_rate": 0.67, "records": [{"round": 1, "choice": "A", "level": 4}]},
    "test_metrics": {"pearson": 0.71, "spearman": 0.68, "regret": 0.09, "distance_to_optimum": 6.4}
  },
  "metrics": {
    "info_gain": [0.21, 0.19, 0.17],
    "posterior_best_mean": [0.41, 0.53, 0.61]
  },
  "metadata": {"mode": "User Study", "n_queries_planned": 40, "n_queries_completed": 40, "status": "complete"}
}
```

## Project Layout
```
.
├── README.md
├── README_zh-CN.md
├── requirements.txt
├── run_study.py
├── run_user_study_ui.py
├── run_auto_test_ui.py
├── xbox_control.py
├── data/
└── src
    └── preference_learning
        ├── __init__.py
        ├── audio
        │   ├── generator.py
        │   └── signal.py
        ├── gp
        │   ├── audio_gp.py
        │   ├── gaussian_process.py
        │   └── math_utils.py
        ├── evaluation.py
        └── interface
            ├── __init__.py
            ├── session.py
            ├── ui_study.py
            └── logo/
```
