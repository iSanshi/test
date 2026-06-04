import argparse
import csv
import pickle
from pathlib import Path

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from contactexplorer_genesis.env import LEAP_CONTACT_STYLE_NAMES, LeapSingulationGenesisEnv


def _mean_metric(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _latest_checkpoint(log_dir: Path) -> Path:
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {log_dir}")
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="leap_singulation_genesis_style_conditioned_v1")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path(args.log_root) / args.run_name)
    cfg_path = ckpt.parent / "cfgs.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfgs.pkl next to checkpoint: {cfg_path}")
    with open(cfg_path, "rb") as f:
        base_env_cfg, reward_cfg, train_cfg = pickle.load(f)

    rows = []
    style_names = list(LEAP_CONTACT_STYLE_NAMES[: getattr(base_env_cfg, "num_contact_styles", 5)])
    for style_id, style_name in enumerate(style_names):
        env_cfg = pickle.loads(pickle.dumps(base_env_cfg))
        env_cfg.num_envs = args.num_envs
        env_cfg.show_viewer = bool(args.viewer)
        env_cfg.use_contact_style_condition = True
        env_cfg.fixed_contact_style = style_id

        env = LeapSingulationGenesisEnv(env_cfg, reward_cfg=reward_cfg)
        runner_cfg = pickle.loads(pickle.dumps(train_cfg))
        runner = OnPolicyRunner(env, runner_cfg, ckpt.parent, device=gs.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=gs.device)

        obs = env.reset()
        metrics = {
            "success": [],
            "goal_dist": [],
            "selected_style_contact": [],
            "style_0_contact": [],
            "style_1_contact": [],
            "style_2_contact": [],
            "style_3_contact": [],
            "style_4_contact": [],
        }
        for _ in range(args.steps):
            with torch.no_grad():
                actions = policy(obs, stochastic_output=args.stochastic)
                if args.action_noise > 0.0:
                    actions = actions + args.action_noise * torch.randn_like(actions)
                actions = torch.clamp(actions, -1.0, 1.0)
            obs, _, _, info = env.step(actions)
            episode = info.get("episode", {})
            for key in metrics:
                value = episode.get(f"rew_{key}")
                if torch.is_tensor(value):
                    metrics[key].append(float(value.detach().cpu()))

        rows.append(
            {
                "style_id": style_id,
                "style_name": style_name,
                "success_rate": _mean_metric(metrics["success"]),
                "final_goal_dist_mean": _mean_metric(metrics["goal_dist"]),
                "selected_style_contact_rate": _mean_metric(metrics["selected_style_contact"]),
                "palm_contact_rate": _mean_metric(metrics["style_1_contact"]),
                "fingertip_contact_rate": _mean_metric(metrics["style_0_contact"]),
                "mcp_contact_rate": _mean_metric(metrics["style_2_contact"]),
                "pip_dip_contact_rate": _mean_metric(metrics["style_3_contact"]),
                "thumb_contact_rate": _mean_metric(metrics["style_4_contact"]),
            }
        )
        del env
        del runner
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = ckpt.parent / "contact_style_eval.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Checkpoint: {ckpt}")
    print(f"Wrote: {out_path}")
    for row in rows:
        print(
            "style={style_id} {style_name}: success={success_rate:.3f}, "
            "goal_dist={final_goal_dist_mean:.3f}, selected_contact={selected_style_contact_rate:.3f}".format(**row)
        )


if __name__ == "__main__":
    main()
