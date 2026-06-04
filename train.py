import argparse
import pickle
from pathlib import Path

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from contactexplorer_genesis.env import (
    LeapSingulationGenesisEnv,
    default_reward_cfg,
    default_train_cfg,
    make_cfg,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp-name", type=str, default="leap_singulation_genesis")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--no-randomize-dynamics", action="store_true")
    parser.add_argument("--state-type", choices=["hash", "predefined"], default="hash")
    parser.add_argument("--hash-code-dim", type=int, default=None)
    parser.add_argument("--hash-hidden-dim", type=int, default=None)
    parser.add_argument("--hash-ae-update-freq", type=int, default=None)
    parser.add_argument("--state-running-max-mode", choices=["global", "state"], default=None)
    parser.add_argument("--reach-curiosity-scale", type=float, default=None)
    parser.add_argument("--contact-coverage-scale", type=float, default=None)
    parser.add_argument("--contact-diversity-scale", type=float, default=None)
    parser.add_argument("--non-fingertip-target-penalty-scale", type=float, default=None)
    parser.add_argument("--action-penalty-scale", type=float, default=None)
    parser.add_argument("--use-ppo-curiosity", action="store_true")
    parser.add_argument(
        "--curiosity-model-type",
        choices=["prediction_error", "rnd", "disagreement", "neural_hash"],
        default="prediction_error",
    )
    parser.add_argument(
        "--curiosity-state-type",
        choices=["state_feature", "policy_state", "contact_force", "contact_distance"],
        default="state_feature",
    )
    parser.add_argument("--intrinsic-reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--curiosity-learning-rate", type=float, default=1e-4)
    parser.add_argument("--curiosity-simhash-dim", type=int, default=5)
    parser.add_argument("--curiosity-code-dim", type=int, default=16)
    parser.add_argument("--wrench-prob", type=float, default=None)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--torque-scale", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    overrides = {
        "num_envs": args.num_envs,
        "seed": args.seed,
        "show_viewer": args.viewer,
        "randomize_dynamics": not args.no_randomize_dynamics,
        "state_type": args.state_type,
        "return_curiosity_info": args.use_ppo_curiosity,
        "curiosity_state_type": args.curiosity_state_type,
    }
    if args.hash_code_dim is not None:
        overrides["hash_code_dim"] = args.hash_code_dim
    if args.hash_hidden_dim is not None:
        overrides["hash_hidden_dim"] = args.hash_hidden_dim
    if args.hash_ae_update_freq is not None:
        overrides["hash_ae_update_freq"] = args.hash_ae_update_freq
    if args.state_running_max_mode is not None:
        overrides["state_running_max_mode"] = args.state_running_max_mode
    if args.wrench_prob is not None:
        overrides["wrench_prob"] = args.wrench_prob
    if args.force_scale is not None:
        overrides["force_scale"] = args.force_scale
    if args.torque_scale is not None:
        overrides["torque_scale"] = args.torque_scale
    env_cfg = make_cfg(**overrides)
    reward_cfg = default_reward_cfg()
    if args.reach_curiosity_scale is not None:
        reward_cfg["reach_curiosity"] = args.reach_curiosity_scale
    if args.contact_coverage_scale is not None:
        reward_cfg["contact_coverage"] = args.contact_coverage_scale
    if args.contact_diversity_scale is not None:
        reward_cfg["contact_diversity"] = args.contact_diversity_scale
    if args.non_fingertip_target_penalty_scale is not None:
        reward_cfg["non_fingertip_target_penalty"] = args.non_fingertip_target_penalty_scale
    if args.action_penalty_scale is not None:
        reward_cfg["action_penalty"] = args.action_penalty_scale
    train_cfg = default_train_cfg(args.exp_name)
    train_cfg["save_interval"] = args.save_interval
    train_cfg["algorithm"]["curiosity_cfg"].update(
        {
            "enabled": args.use_ppo_curiosity,
            "model_type": args.curiosity_model_type,
            "intrinsic_reward_scale": args.intrinsic_reward_scale,
            "learning_rate": args.curiosity_learning_rate,
            "reward_scale": args.reward_scale,
            "simhash_dim": args.curiosity_simhash_dim,
            "code_dim": args.curiosity_code_dim,
        }
    )
    log_dir = Path(args.log_root) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_cfg, train_cfg), f)

    env = LeapSingulationGenesisEnv(env_cfg, reward_cfg=reward_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume_checkpoint is not None:
        runner.load(args.resume_checkpoint, map_location=str(gs.device))
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
