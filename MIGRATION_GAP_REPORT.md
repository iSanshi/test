# Genesis Migration Gap Report

This report compares the current Genesis migration in `/mnt/p5/contactexplorer_genesis`
against the original ContactExplorer Isaac Gym code under `/mnt/p5/ContactExplorer/repo`.

## Implemented In The Genesis Migration

- LEAP singulation scene:
  - xArm6 + LEAP hand URDF
  - table
  - 5 target/neighbor boxes
  - goal marker
- Genesis batched RL environment:
  - `reset`
  - `step`
  - TensorDict observation for RSL-RL
  - checkpoint training/evaluation scripts
- Real Genesis contact integration:
  - target-tip contact
  - target-table contact
  - target-neighbor contact
  - non-target robot/arm contact
  - contact position and contact force logging
- Contact Coverage-Guided Exploration environment reward:
  - canonical cube surface grid and canonical cube normals
  - object-orientation-aware world surface point clouds
  - contact positions transformed back into target local frame
  - farthest-point-sampled state points and cluster initialization
  - normal-aware surface clustering
  - backface and palm-inward surface masks
  - inverse/gaussian/exponential potential kernels
  - exponential/linear/sqrt/logarithmic novelty decay
  - state/keypoint/cluster contact counters
  - state/global running-max potential shaping
  - learned hash state bank with hash autoencoder + SimHash
  - predefined progress-bin state bank fallback
- ContactExplorer LEAP singulation PPO defaults:
  - actor/critic hidden sizes `[512, 256, 128]`
  - observation normalization enabled
  - value running mean/std normalization enabled
  - reward scale `0.01`
  - PPO clip range `0.1`, entropy coefficient `0`, rollout steps `12`,
    learning epochs `2`
  - `successSteps=60`
- PPO-side intrinsic curiosity:
  - prediction error
  - RND
  - disagreement ensemble
  - neural hash
  - `next_curiosity_states` rollout storage
  - intrinsic reward addition during rollout
  - curiosity model training during PPO update
- Genesis domain randomization/disturbance:
  - object friction ratio
  - object mass shift
  - object COM shift
  - robot PD gain randomization
  - random external force/torque on the target

## Remaining Differences

### Backend Physics

- Original uses Isaac Gym PhysX tensor APIs; migration uses Genesis rigid solver.
- Contact force semantics, solver tolerances, collision pair generation, and link
  inertia handling are not byte-identical.
- Genesis currently prints a legacy URDF inertia warning for the xArm/LEAP URDF.

### Task Coverage

The original repository registers many tasks:

- XArm + Allegro singulation/table-top/cube-in-box
- XArm + LEAP singulation/table-top/cube-in-box
- In-hand Allegro
- In-hand LEAP
- Bimanual LEAP articulation/board lift
- PushBox2D
- xArm7 LEAP variants

The Genesis migration currently ports only the LEAP singulation workflow.

### Scene And Object Data

- Original tasks can use dataset-backed object point clouds, normals, metadata,
  categories, bounding boxes, and task-specific object layouts.
- Current Genesis singulation uses fixed cube boxes and a procedural cube surface
  grid. It does not load the full object datasets.
- For this cube singulation scene the canonical cube point cloud and normals are
  now procedurally aligned to the original CCGE logic, but arbitrary
  dataset-backed objects are still not ported.

### Observations

- Original observation spaces are Hydra-configured and can include stacked frames,
  task-specific observation specs, keypoint sets, object metadata, and optional
  curiosity observations.
- Current Genesis policy observation is a compact 96-D vector:
  robot qpos/qvel, palm pose proxy, target pose, goal delta, fingertip-relative
  positions, nearest-neighbor distance, contact flags/counts, and last actions.
- Current Genesis curiosity states support `state_feature`, `policy_state`,
  `contact_force`, and `contact_distance`, but not every original observation
  spec file.
- Observation and value normalization are enabled to match the original LEAP
  PPO config, but the exact observation vector is still not byte-identical.

### Reward And Reset Details

- Current Genesis reward keeps the core singulation terms and the CCGE scalings:
  target progress, reach, reach curiosity, contact coverage, neighbor stability,
  action penalty, and success.
- Original singulation code has more Isaac-specific and task-specific terms,
  gates, object displacement checks, Jacobian/singularity diagnostics, action
  noise tracking, object masks, and dataset-dependent bookkeeping.
- Current success/failure logic uses the original 60 consecutive near-goal steps,
  but failure conditions are still simplified to target fall and non-target arm
  contact.

### Control

- Current Genesis uses Genesis IK for the first 6 arm action dimensions and
  position targets for LEAP fingers.
- Original Isaac task has its own control pipeline, tensor refresh order, DOF
  indexing, and potentially different arm/hand controller behavior.

### Evaluation And Visualization

- Current viewer/eval scripts run random actions or trained policy playback.
- Original paper/demo videos may include policies, camera/render settings, and
  task variants that are not present in this Genesis-only LEAP singulation port.

## Practical Status

The migration now covers the main algorithmic idea:

1. Contact coverage as an environment reward.
2. Learned hash autoencoder state discretization.
3. PPO-side intrinsic curiosity model.
4. Original LEAP singulation PPO hyperparameters that are simulator-independent.

It should be treated as a functional Genesis prototype for LEAP singulation, not
as a byte-identical reproduction of all ContactExplorer experiments. The largest
remaining mismatch is no longer the CCGE/PPO method itself; it is the simulator
backend, exact observation/control tensor pipeline, and full multi-task/object
dataset coverage.
