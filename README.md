# haptic-preference-learning

[中文版](README_zh-CN.md)

This repo provides a **preference-based haptic personalization** framework that learns a user’s latent utility from **binary A/B choices**. We use a **Gaussian Process (GP) preference model** to capture smoothness and uncertainty over the stimulus space, and an **active query policy** that maximizes **expected information gain** to pick the next comparison. Users can report **response uncertainty**, which is used as per-comparison weights to down-weight ambiguous judgments. By emphasizing **relative** (not absolute) evaluations, the system reduces rating fatigue and drift and avoids forcing tactile sensations onto a numeric scale.

**Highlights**
- GP preference learning over haptic stimuli (uncertainty-aware, smoothness prior)  
- Information-gain active querying for sample-efficient searches  
- Per-comparison **uncertainty weighting** to handle ambiguous answers  
- Open, extensible code for interactive preference search

![UI Demo](image.png)

## Quick Start
1. Clone the repo:
   ```bash
   git clone https://github.com/iSanshi/haptic-preference-learning.git
   cd haptic-preference-learning
   ```
2. (Optional) create a virtual environment:

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the UI:
   ```bash
   python run_user_study_ui.py   # manual study workflow
   python run_auto_test_ui.py    # automated testing workflow
   ```

5. Press **Begin** in the UI. In user mode you manually pick A/B clips and rate certainty (1–5); in auto-test mode the system simulates preferences via a ground-truth function.
6. When a session completes, results are exported to `data/YYYYMMDD_<index>/session.json` and `log.txt`.

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
    "bounds": {"amplitude": [20.0, 100.0], "frequency": [20.0, 100.0], "density": [20.0, 100.0], "gradient": [20.0, 100.0]},
    "posterior_uncertainty": {"avg_pred_var": 0.12, "max_pred_var": 0.41},
    "validation": {"rounds": 3, "win_rate": 0.67, "records": [{"round": 1, "choice": "A", "level": 4}]},
    "test_metrics": {"pearson": 0.71, "spearman": 0.68, "regret": 0.09, "distance_to_optimum": 6.4}
  },
  "metrics": {
    "info_gain": [0.21, 0.19, 0.17],
    "posterior_best_mean": [0.41, 0.53, 0.61]
  },
  "metadata": {"mode": "User Study", "n_queries_planned": 35, "n_queries_completed": 35, "status": "complete"}
}
```

## Project Layout
```
.
├── requirements.txt
├── README.md
├── README.zh-CN.md
├── run_user_study_ui.py
├── run_auto_test_ui.py
├── tutorial
    ├── gp_interactive_Chinese.html
    └── gp_interactive_English.html
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
