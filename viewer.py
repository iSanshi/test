import argparse
import time

import torch
import genesis as gs

from contactexplorer_genesis.env import LeapSingulationGenesisEnv, make_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-randomize-dynamics", action="store_true")
    parser.add_argument("--state-type", choices=["hash", "predefined"], default="hash")
    parser.add_argument("--wrench-prob", type=float, default=None)
    args = parser.parse_args()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed)

    overrides = {
        "num_envs": args.num_envs,
        "seed": args.seed,
        "show_viewer": not args.headless,
        "randomize_dynamics": not args.no_randomize_dynamics,
        "state_type": args.state_type,
    }
    if args.wrench_prob is not None:
        overrides["wrench_prob"] = args.wrench_prob
    env = LeapSingulationGenesisEnv(make_cfg(**overrides))
    for _ in range(args.steps):
        if args.random_actions:
            actions = torch.empty((env.num_envs, env.num_actions), device=gs.device).uniform_(-0.25, 0.25)
        else:
            actions = torch.zeros((env.num_envs, env.num_actions), device=gs.device)
        env.step(actions)
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
