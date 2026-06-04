import argparse
import pickle
import time
from pathlib import Path

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from contactexplorer_genesis.env import LeapSingulationGenesisEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=0,
        help="Reset all viewer environments every N steps to show new randomized poses. Disabled when <= 0.",
    )
    args = parser.parse_args()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    ckpt = Path(args.checkpoint)
    cfg_path = ckpt.parent / "cfgs.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfgs.pkl next to checkpoint: {cfg_path}")
    with open(cfg_path, "rb") as f:
        env_cfg, reward_cfg, train_cfg = pickle.load(f)
    env_cfg.num_envs = args.num_envs
    env_cfg.show_viewer = args.viewer and not args.headless

    env = LeapSingulationGenesisEnv(env_cfg, reward_cfg=reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, ckpt.parent, device=gs.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    for step in range(args.steps):
        if args.reset_interval > 0 and step > 0 and step % args.reset_interval == 0:
            obs = env.reset()
        with torch.no_grad():
            actions = policy(obs, stochastic_output=args.stochastic)
            if args.action_noise > 0.0:
                actions = actions + args.action_noise * torch.randn_like(actions)
            actions = torch.clamp(actions, -1.0, 1.0)
        obs, _, _, _ = env.step(actions)
        if args.sleep > 0.0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
