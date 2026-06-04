# ContactExplorer LEAP Singulation on Genesis

This is a migration attempt for the ContactExplorer `leap_singulation` task from
Isaac Gym to Genesis World.

It is not a byte-for-byte port of the Isaac Gym task. The original task is tightly
coupled to Isaac Gym/PhysX tensor APIs, so the environment is reimplemented with
Genesis primitives and Genesis batched simulation.

## What Is Migrated

- Genesis scene with:
  - xArm6 + LEAP hand URDF from ContactExplorer assets
  - table
  - 5 singulation boxes
  - target marker
- Gym-like RL environment:
  - `reset()`
  - `step(actions)`
  - observation TensorDict for RSL-RL
  - reward, done, extras
- Genesis contact integration:
  - reads `scene.rigid_solver.collider.get_contacts()`
  - logs fingertip-target, target-table, target-neighbor, and detach/contact flags
  - adds contact features to policy observations
- Genesis IK integration:
  - first 6 action dimensions move and rotate the xArm end-effector target through Genesis IK
  - remaining 16 action dimensions control LEAP finger joints
- Episode termination:
  - configurable near-goal `success_steps`
  - resets on target fall
  - resets on non-target arm contact while allowing normal hand/table contact
- Domain randomization and disturbances:
  - per-env object friction ratio
  - per-env object mass shift
  - per-env object COM shift
  - per-env robot PD gain randomization
  - random external force/torque wrench on the target object
- Contact coverage exploration:
  - rotates the canonical cube point cloud and normals by the target object's
    current pose before computing state features and surface distances
  - maps fingertip-target contact positions back into the target local frame
    before nearest-surface assignment
  - uses a state-conditioned counter shaped like the original task:
    `state_id x fingertip/keypoint x surface_cluster`
  - uses farthest-point sampling for state points and cluster initialization
  - supports normal-aware clustering with canonical cube normals
  - supports backface and palm-inward contact masks
  - supports inverse/gaussian/exponential potential kernels and
    exponential/linear/sqrt/logarithmic novelty decay modes
  - supports a Genesis-native learned hash state bank:
    surface point-cloud features -> hash autoencoder -> SimHash state id
  - supports a `predefined` progress-bin state bank as a debugging fallback
  - uses running-max shaping for potential and contact novelty rewards
  - logs `contact_count`, `cluster_novelty_reward`, `avg_potential`,
    `stateid_entropy`, `hash_recon_loss`, `hash_binary_reg`, `state_coverage`,
    and per-keypoint contact fields
- Random-action viewer
- PPO training entry using Genesis' RSL-RL style
- ContactExplorer-style PPO defaults for this LEAP singulation port:
  - actor/critic MLP hidden sizes `[512, 256, 128]`
  - observation normalization enabled
  - value running mean/std normalization enabled and saved in checkpoints
  - PPO clip range `0.1`, entropy coefficient `0`, rollout steps `12`,
    learning epochs `2`
  - environment reward scale `0.01`
  - `success_steps=60`
- ContactExplorer-style PPO-side intrinsic curiosity:
  - prediction error
  - RND
  - disagreement ensemble
  - neural hash
  - stores `next_curiosity_states`, adds intrinsic reward during rollout, and
    trains the curiosity model during PPO updates
- Checkpoint evaluation viewer

## What Is Not Yet Equivalent To The Paper

- ContactExplorer's exact Isaac Gym net-contact-force tensor semantics are not replicated.
- The original `CuriosityRewardManager` is not imported byte-for-byte because it
  depends on Isaac Gym project modules, but its learned hash state-bank behavior
  and the CCGE surface/normal/state-counter logic are ported as pure PyTorch
  Genesis-native modules.
- ContactExplorer's PPO-side intrinsic curiosity is implemented as a local
  `CuriosityPPO` wrapper around RSL-RL PPO. It matches the algorithmic data flow,
  but it is not a byte-for-byte copy of the original Isaac Gym PPO runner.
- Domain randomization for scale and some Isaac-specific random wrench details is
  not byte-for-byte ported, but Genesis mass/COM/friction/PD/wrench randomization
  is implemented.
- The dense reward keeps the core singulation signals and CCGE reward scalings,
  but Isaac-specific diagnostics/gates that depend on Isaac tensor semantics are
  still not byte-identical.
- Physics/contact behavior will not match Isaac Gym PhysX exactly.

This should be treated as a Genesis backend prototype, not a paper-reproduction
replacement.

## Requirements

Genesis World must be installed or runnable from `/mnt/p5/genesis-world-v1.0.0`.
This migration has a local virtual environment at:

```text
/mnt/p5/contactexplorer_genesis/.venv
```

It was created with GPU PyTorch and the local Genesis source:

```bash
python3.12 -m venv /mnt/p5/contactexplorer_genesis/.venv
/mnt/p5/contactexplorer_genesis/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
/mnt/p5/contactexplorer_genesis/.venv/bin/pip install -e /mnt/p5/genesis-world-v1.0.0 rsl-rl-lib tensordict
```

## Commands

From this directory:

```bash
./scripts/viewer.sh --backend gpu --num-envs 1 --random-actions
./scripts/viewer.sh --backend gpu --num-envs 1 --random-actions --headless --steps 2

./scripts/train.sh --backend gpu --num-envs 256 --max-iterations 1000 --save-interval 100
./scripts/train_leap_singulation_learn.sh
./scripts/train.sh --backend gpu --num-envs 256 --max-iterations 1000 --wrench-prob 0.02 --force-scale 0.6 --torque-scale 0.02
./scripts/train.sh --backend gpu --num-envs 256 --max-iterations 1000 --no-randomize-dynamics
./scripts/train.sh --backend gpu --num-envs 256 --max-iterations 1000 --state-type hash
./scripts/train.sh --backend gpu --num-envs 256 --max-iterations 1000 --state-type predefined
./scripts/train.sh --backend gpu --num-envs 256 --max-iterations 1000 --use-ppo-curiosity --curiosity-model-type prediction_error --intrinsic-reward-scale 1.0
./scripts/train.sh --backend gpu --num-envs 256 --max-iterations 1000 --use-ppo-curiosity --curiosity-model-type neural_hash --curiosity-state-type state_feature
./scripts/eval.sh --backend gpu --checkpoint logs/leap_singulation_genesis/model_1000.pt --num-envs 1 --viewer
```

If you already have Genesis installed in another Python environment:

```bash
PYTHONPATH=/mnt/p5/contactexplorer_genesis python viewer.py --random-actions
```

## Asset Path

By default the code reads assets from:

```text
/mnt/p5/ContactExplorer/repo/assets
```

Override with:

```bash
export CONTACTEXPLORER_ASSETS=/path/to/ContactExplorer/repo/assets
```
