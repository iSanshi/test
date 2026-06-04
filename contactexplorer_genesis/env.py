from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tensordict import TensorDict

import genesis as gs

from contactexplorer_genesis.hash_state_bank import LearnedHashStateBank


LEAP_FINGERTIP_LINKS = [
    "fingertip",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
]

LEAP_FINGERTIP_CONTACT_LINK_GROUPS = [
    ("fingertip", "index_tip_head"),
    ("fingertip_2", "middle_tip_head"),
    ("fingertip_3", "ring_tip_head"),
    ("thumb_fingertip", "thumb_tip_head"),
]

LEAP_CONTACT_DIVERSITY_LINKS = [
    "palm_lower",
    "mcp_joint",
    "pip",
    "dip",
    "fingertip",
    "index_tip_head",
    "mcp_joint_2",
    "pip_2",
    "dip_2",
    "fingertip_2",
    "middle_tip_head",
    "mcp_joint_3",
    "pip_3",
    "dip_3",
    "fingertip_3",
    "ring_tip_head",
    "thumb_temp_base",
    "thumb_pip",
    "thumb_dip",
    "thumb_fingertip",
    "thumb_tip_head",
]


@dataclass
class LeapSingulationGenesisConfig:
    num_envs: int = 64
    num_objects: int = 5
    num_actions: int = 22
    ctrl_dt: float = 1.0 / 60.0
    control_decimation: int = 4
    episode_length: int = 300
    episode_length_s: float = 4.0
    actions_moving_average: float = 0.8
    action_scale: float = 0.035
    arm_pos_action_scale: float = 0.035
    arm_rot_action_scale: float = 0.15
    finger_action_scale: float = 0.035
    use_arm_ik: bool = True
    enable_contact_coverage: bool = True
    coverage_grid: int = 4
    num_key_states: int = 32
    surface_cluster_k: int = 32
    state_type: str = "hash"
    state_num_points: int = 32
    state_include_goal: bool = True
    state_running_max_mode: str = "global"
    hash_code_dim: int = 256
    hash_hidden_dim: int = 512
    hash_noise_scale: float = 0.3
    hash_lambda_binary: float = 10.0
    hash_ae_lr: float = 3e-4
    hash_ae_steps: int = 5
    hash_ae_update_freq: int = 16
    hash_ae_num_minibatches: int = 8
    hash_seed: int = 0
    return_curiosity_info: bool = False
    curiosity_state_type: str = "state_feature"
    potential_kernel: str = "exponential"
    novelty_decay: str = "sqrt"
    novelty_decay_rate: float = 2.0
    curiosity_kernel_param: float = 0.03
    use_potential_shaping: bool = True
    use_normal_in_clustering: bool = False
    normal_weight: float = 0.5
    max_clustering_iters: int = 10
    mask_backface_points: bool = True
    mask_palm_inward_points: bool = True
    contact_force_threshold: float = 1e-3
    arm_contact_force_threshold: float = 1.0
    require_fingertip_target_contact: bool = False
    near_surface_threshold: float = 0.012
    success_steps: int = 60
    fail_on_target_fall: bool = True
    target_fall_z_margin: float = 0.10
    fail_on_non_target_robot_contact: bool = True
    randomize_dynamics: bool = True
    randomize_friction: bool = True
    friction_ratio_range: tuple[float, float] = (0.75, 1.50)
    randomize_mass: bool = True
    mass_shift_range: tuple[float, float] = (-0.015, 0.030)
    randomize_com: bool = True
    com_shift_range: tuple[float, float] = (-0.006, 0.006)
    randomize_pd_gains: bool = True
    p_gain_range: tuple[float, float] = (350.0, 650.0)
    d_gain_range: tuple[float, float] = (25.0, 50.0)
    wrench_prob: float = 0.02
    force_scale: float = 0.6
    torque_scale: float = 0.02
    table_size: tuple[float, float, float] = (0.8, 0.8, 0.04)
    table_pos: tuple[float, float, float] = (0.0, 0.0, 0.35)
    box_size: tuple[float, float, float] = (0.055, 0.055, 0.055)
    object_z: float = 0.40
    goal_pos: tuple[float, float, float] = (0.0, 0.28, 0.40)
    goal_dist_min_initial: float = 0.60
    success_dist: float = 0.075
    arm_hand_urdf: str = "urdf/xarm6_leap_vertical_moving.urdf"
    arm_base_pos: tuple[float, float, float] = (0.0, 0.55, 0.0)
    # ContactExplorer xyzw [0, 0, -sqrt(.5), sqrt(.5)] converted to Genesis wxyz.
    arm_base_quat: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, -0.70710678)
    palm_link: str = "palm_lower"
    eef_link: str = "link6"
    seed: int = 1
    show_viewer: bool = False
    randomize_object_positions: bool = True


def default_assets_dir() -> Path:
    return Path(os.environ.get("CONTACTEXPLORER_ASSETS", "/mnt/p5/ContactExplorer/repo/assets"))


def make_cfg(**overrides) -> LeapSingulationGenesisConfig:
    cfg = LeapSingulationGenesisConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown env config key: {key}")
        setattr(cfg, key, value)
    return cfg


class LeapSingulationGenesisEnv:
    """Genesis reimplementation of the core LEAP singulation workflow.

    This environment intentionally keeps a small, explicit state/action surface.
    It is designed to be trainable and inspectable first, while preserving the
    main contact-state coverage structure used by ContactExplorer.
    """

    def __init__(self, cfg: LeapSingulationGenesisConfig, reward_cfg: dict[str, float] | None = None):
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.num_actions = int(cfg.num_actions)
        self.device = gs.device
        self.control_decimation = max(1, int(getattr(cfg, "control_decimation", 1)))
        self.max_episode_length = int(getattr(cfg, "episode_length", 0))
        if self.max_episode_length <= 0:
            control_dt = cfg.ctrl_dt * self.control_decimation
            self.max_episode_length = math.ceil(cfg.episode_length_s / control_dt)
        self.reward_cfg = reward_cfg or default_reward_cfg()
        self.assets_dir = default_assets_dir()

        urdf_file = self.assets_dir / cfg.arm_hand_urdf
        if not urdf_file.exists():
            raise FileNotFoundError(f"Missing xArm+LEAP URDF: {urdf_file}")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=cfg.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_links_info=True,
                batch_dofs_info=True,
                max_collision_pairs=512,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(4, self.num_envs))),
                env_separate_rigid=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=(1.2, -1.2, 1.1),
                camera_lookat=(0.0, 0.1, 0.35),
                camera_fov=48,
                max_FPS=60,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=cfg.show_viewer,
        )

        self._build_scene(urdf_file)
        self._init_buffers()
        self.reset()

    def _build_scene(self, urdf_file: Path) -> None:
        self.scene.add_entity(gs.morphs.Plane())

        self.table = self.scene.add_entity(
            gs.morphs.Box(pos=self.cfg.table_pos, size=self.cfg.table_size, fixed=True),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.45, 0.45, 0.45))),
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf_file),
                pos=self.cfg.arm_base_pos,
                quat=self.cfg.arm_base_quat,
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
                links_to_keep=tuple([self.cfg.palm_link, self.cfg.eef_link, *LEAP_FINGERTIP_LINKS]),
            )
        )

        self.objects = []
        colors = [
            (0.95, 0.95, 0.95),
            (0.90, 0.15, 0.10),
            (0.95, 0.55, 0.05),
            (0.20, 0.45, 0.90),
            (0.15, 0.70, 0.30),
        ]
        for i in range(self.cfg.num_objects):
            obj = self.scene.add_entity(
                gs.morphs.Box(size=self.cfg.box_size, fixed=False, batch_fixed_verts=True),
                surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=colors[i % len(colors)])),
            )
            self.objects.append(obj)

        self.goal_marker = self.scene.add_entity(
            gs.morphs.Box(pos=self.cfg.goal_pos, size=(0.08, 0.08, 0.012), fixed=True, collision=False),
            surface=gs.surfaces.Default(diffuse_texture=gs.textures.ColorTexture(color=(0.2, 0.7, 1.0, 0.35))),
        )

        self.scene.build(n_envs=self.num_envs, env_spacing=(1.0, 1.0))

        self._setup_robot_handles()
        self._setup_link_indices()
        self._setup_contact_sensors()
        self._setup_robot_control()

    def _setup_robot_handles(self) -> None:
        self.palm_link = self._get_link_or_none(self.cfg.palm_link)
        self.eef_link = self._get_link_or_none(self.cfg.eef_link)
        self.fingertip_links = [link for link in (self._get_link_or_none(name) for name in LEAP_FINGERTIP_LINKS) if link]
        self.fingertip_contact_link_groups = [
            [link for link in (self._get_link_or_none(name) for name in group) if link]
            for group in LEAP_FINGERTIP_CONTACT_LINK_GROUPS
        ]
        self.contact_diversity_links = [
            link for link in (self._get_link_or_none(name) for name in LEAP_CONTACT_DIVERSITY_LINKS) if link
        ]
        self.robot_q_dim = int(getattr(self.robot, "n_qs", getattr(self.robot, "n_dofs", self.num_actions)))
        self.robot_dof_dim = int(getattr(self.robot, "n_dofs", self.robot_q_dim))
        self.control_dim = min(self.num_actions, self.robot_q_dim)
        self.num_actions = self.control_dim

        default = [0.0, -1.0, -0.5, 0.0, 0.0, 0.0]
        default += [
            0.0,
            0.95,
            0.66,
            0.80,
            0.0,
            0.95,
            0.66,
            0.80,
            0.0,
            0.95,
            0.66,
            0.80,
            0.85,
            0.95,
            0.30,
            0.24,
        ]
        if len(default) < self.robot_q_dim:
            default += [0.0] * (self.robot_q_dim - len(default))
        self.default_qpos = torch.tensor(default[: self.robot_q_dim], dtype=gs.tc_float, device=self.device)
        self.current_targets = self.default_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        self.arm_dofs_idx = list(range(min(6, self.robot_dof_dim)))
        self.finger_dofs_idx = list(range(6, self.control_dim))
        self.default_eef_quat = self._link_quat(self.eef_link)
        self.current_eef_quat = self.default_eef_quat.clone()

    def _setup_link_indices(self) -> None:
        self.target_link_idx = int(self.objects[0].links[0].idx)
        self.neighbor_link_indices = torch.tensor(
            [int(obj.links[0].idx) for obj in self.objects[1:]],
            dtype=gs.tc_int,
            device=self.device,
        )
        self.table_link_idx = int(self.table.links[0].idx)
        self.fingertip_link_indices = torch.tensor(
            [int(link.idx) for link in self.fingertip_links],
            dtype=gs.tc_int,
            device=self.device,
        )
        fingertip_contact_ids = sorted(
            {
                int(link.idx)
                for group in self.fingertip_contact_link_groups
                for link in group
            }
        )
        self.fingertip_contact_link_indices = torch.tensor(
            fingertip_contact_ids,
            dtype=gs.tc_int,
            device=self.device,
        )
        self.fingertip_contact_group_indices = [
            torch.tensor([int(link.idx) for link in group], dtype=gs.tc_int, device=self.device)
            for group in self.fingertip_contact_link_groups
        ]
        self.contact_diversity_link_indices = torch.tensor(
            [int(link.idx) for link in self.contact_diversity_links],
            dtype=gs.tc_int,
            device=self.device,
        )
        self.robot_link_indices = torch.tensor(
            [int(link.idx) for link in self.robot.links],
            dtype=gs.tc_int,
            device=self.device,
        )
        hand_link_ids = {
            int(link.idx)
            for link in self.robot.links
            if not getattr(link, "name", "").startswith("link")
        }
        if self.palm_link is not None:
            hand_link_ids.add(int(self.palm_link.idx))
        self.hand_link_indices = torch.tensor(
            sorted(hand_link_ids),
            dtype=gs.tc_int,
            device=self.device,
        )
        self.arm_link_indices = torch.tensor(
            [int(link.idx) for link in self.robot.links if int(link.idx) not in hand_link_ids],
            dtype=gs.tc_int,
            device=self.device,
        )

    def _setup_contact_sensors(self) -> None:
        self.contact_sensors = []
        if not hasattr(gs, "sensors"):
            return
        for link in self.fingertip_links:
            try:
                sensor = self.scene.add_sensor(
                    gs.sensors.ContactForce(entity_idx=self.robot.idx, link_idx_local=link.idx_local)
                )
                self.contact_sensors.append(sensor)
            except Exception:
                # Sensor API availability differs across Genesis versions.
                pass

    def _setup_robot_control(self) -> None:
        kp = torch.full((self.robot_dof_dim,), 500.0, dtype=gs.tc_float, device=self.device)
        kv = torch.full((self.robot_dof_dim,), 35.0, dtype=gs.tc_float, device=self.device)
        force_low = torch.full((self.robot_dof_dim,), -120.0, dtype=gs.tc_float, device=self.device)
        force_high = torch.full((self.robot_dof_dim,), 120.0, dtype=gs.tc_float, device=self.device)
        try:
            self.robot.set_dofs_kp(kp)
            self.robot.set_dofs_kv(kv)
            self.robot.set_dofs_force_range(force_low, force_high)
        except Exception:
            pass
        self.base_kp = kp
        self.base_kv = kv

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=gs.tc_int, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=self.device)
        self.prev_actions = torch.zeros_like(self.last_actions)
        self.goal_pos = torch.tensor(self.cfg.goal_pos, dtype=gs.tc_float, device=self.device).repeat(self.num_envs, 1)
        self.initial_target_pos = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.initial_neighbor_pos = torch.zeros((self.num_envs, self.cfg.num_objects - 1, 3), dtype=gs.tc_float, device=self.device)
        self.goal_dist_min = torch.full((self.num_envs,), 1.0, dtype=gs.tc_float, device=self.device)
        self.keypoints_to_surface_dist_min = torch.full(
            (self.num_envs, len(LEAP_FINGERTIP_LINKS)),
            0.30,
            dtype=gs.tc_float,
            device=self.device,
        )
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=gs.tc_int, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        self.failed = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=self.device)
        self.last_wrench_force_norm = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        self.last_wrench_torque_norm = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        self.canonical_surface_points = self._make_cube_surface_points()
        self.canonical_surface_normals = self._make_cube_surface_normals()
        self.surface_point_to_cluster = self._assign_surface_clusters()
        self.state_point_indices = self._make_state_point_indices()
        self.curiosity_state_dim = self._infer_curiosity_state_dim()
        self.coverage_counts = torch.zeros(
            (self.num_envs, self.canonical_surface_points.shape[0]),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.episode_fingertip_contact_counts = torch.zeros(
            (self.num_envs, len(LEAP_FINGERTIP_LINKS)),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.episode_fingertip_cluster_counts = torch.zeros(
            (self.num_envs, len(LEAP_FINGERTIP_LINKS), self.cfg.surface_cluster_k),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.episode_contact_link_counts = torch.zeros(
            (self.num_envs, len(self.contact_diversity_links)),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.episode_contact_link_cluster_counts = torch.zeros(
            (self.num_envs, len(self.contact_diversity_links), self.cfg.surface_cluster_k),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.global_coverage_counts = torch.zeros(
            self.canonical_surface_points.shape[0],
            dtype=gs.tc_float,
            device=self.device,
        )
        self.state_contact_counts = torch.zeros(
            (self.cfg.num_key_states, len(LEAP_FINGERTIP_LINKS), self.cfg.surface_cluster_k),
            dtype=gs.tc_float,
            device=self.device,
        )
        self.hash_state_bank = self._make_hash_state_bank()
        if self.hash_state_bank is not None:
            self.state_contact_counts = self.hash_state_bank.counts
        running_max_shape = (
            (self.num_envs, self.cfg.num_key_states, len(LEAP_FINGERTIP_LINKS))
            if self.cfg.state_running_max_mode == "state"
            else (self.num_envs, len(LEAP_FINGERTIP_LINKS))
        )
        self.potential_per_kp_max = torch.zeros(running_max_shape, dtype=gs.tc_float, device=self.device)
        self.contact_coverage_per_kp_max = torch.zeros_like(self.potential_per_kp_max)
        self.contact_info = self._empty_contact_info()
        self.extras = {"episode": {}}

    def reset(self) -> TensorDict:
        self._reset_idx(None)
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        if actions.shape[-1] != self.num_actions:
            raise ValueError(f"Expected actions dim {self.num_actions}, got {actions.shape[-1]}")

        act_moving_average = float(getattr(self.cfg, "actions_moving_average", 1.0))
        smoothed_actions = act_moving_average * actions + (1.0 - act_moving_average) * self.prev_actions
        self.prev_actions[:] = smoothed_actions

        full_target = self._compute_control_targets(smoothed_actions)
        self.current_targets = full_target
        self.robot.control_dofs_position(position=full_target)
        self._maybe_apply_random_wrench()

        for _ in range(self.control_decimation):
            self.scene.step()
        self.episode_length_buf += 1

        reward, reward_terms = self._compute_reward(smoothed_actions)
        done = self.episode_length_buf >= self.max_episode_length
        if hasattr(self.scene, "rigid_solver"):
            try:
                done |= self.scene.rigid_solver.get_error_envs_mask()
            except Exception:
                pass
        done |= reward_terms["success"].bool()
        done |= reward_terms["failed"].bool()

        extras = {
            "time_outs": (self.episode_length_buf >= self.max_episode_length).float(),
            "episode": self._episode_log_terms(reward_terms),
        }
        if self.cfg.return_curiosity_info:
            extras["curiosity_states"] = self._current_curiosity_states().detach()
        self.extras = extras

        self.last_actions[:] = smoothed_actions
        if done.any():
            self._reset_idx(done)
        return self.get_observations(), reward, done, self.extras

    def get_observations(self) -> TensorDict:
        qpos = self._safe_tensor(self.robot.get_qpos, (self.num_envs, self.robot_q_dim))
        qvel = self._safe_tensor(getattr(self.robot, "get_qvel", None), (self.num_envs, self.robot_q_dim))
        target_pos = self.objects[0].get_pos()
        target_quat = self.objects[0].get_quat()
        palm_pos = self._link_pos(self.palm_link)
        tip_pos = self._fingertip_positions()
        rel_tips = (tip_pos - target_pos[:, None, :]).reshape(self.num_envs, -1)
        neighbor_pos = torch.stack([obj.get_pos() for obj in self.objects[1:]], dim=1)
        nearest_neighbor_dist = torch.norm(neighbor_pos - target_pos[:, None, :], dim=-1).min(dim=1).values
        contact_features = torch.cat(
            [
                self.contact_info["hand_target_contact"].float().unsqueeze(-1),
                self.contact_info["table_contact"].float().unsqueeze(-1),
                self.contact_info["env_detach"].float().unsqueeze(-1),
                self.contact_info["hand_contact_count"].unsqueeze(-1)
                / max(float(len(self.contact_diversity_links)), 1.0),
            ],
            dim=-1,
        )
        obs = torch.cat(
            [
                qpos,
                qvel * 0.1,
                palm_pos,
                target_pos,
                target_quat,
                self.goal_pos - target_pos,
                rel_tips,
                nearest_neighbor_dist.unsqueeze(-1),
                contact_features,
                self.last_actions,
            ],
            dim=-1,
        )
        self.obs_buf = obs
        self.num_obs = int(obs.shape[-1])
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx: torch.Tensor | None) -> None:
        if envs_idx is None:
            envs_idx = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if envs_idx.dtype != torch.bool:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[envs_idx] = True
            envs_idx = mask

        n = int(envs_idx.sum().item())
        if n == 0:
            return

        self.robot.set_qpos(self.default_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)
        self.current_targets[envs_idx] = self.default_qpos

        base = torch.tensor(
            [
                [-0.06, 0.00, self.cfg.object_z],
                [0.00, 0.00, self.cfg.object_z],
                [0.06, 0.00, self.cfg.object_z],
                [-0.03, -0.06, self.cfg.object_z],
                [0.03, -0.06, self.cfg.object_z],
            ],
            dtype=gs.tc_float,
            device=self.device,
        )[: self.cfg.num_objects]
        if self.cfg.randomize_object_positions:
            noise = torch.zeros((n, self.cfg.num_objects, 3), dtype=gs.tc_float, device=self.device)
            noise[:, :, :2] = (torch.rand((n, self.cfg.num_objects, 2), device=self.device) - 0.5) * 0.035
        else:
            noise = torch.zeros((n, self.cfg.num_objects, 3), dtype=gs.tc_float, device=self.device)
        obj_pos = base.unsqueeze(0) + noise

        env_ids = envs_idx.nonzero(as_tuple=False).flatten()
        for i, obj in enumerate(self.objects):
            pos_i = obj_pos[:, i, :]
            obj.set_pos(pos_i, envs_idx=env_ids, skip_forward=True)
            obj.set_quat(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=self.device).repeat(n, 1),
                envs_idx=env_ids,
                skip_forward=i != len(self.objects) - 1,
            )
        self._apply_domain_randomization(env_ids)

        self.initial_target_pos[envs_idx] = obj_pos[:, 0, :]
        if self.cfg.num_objects > 1:
            self.initial_neighbor_pos[envs_idx] = obj_pos[:, 1:, :]
        self.goal_dist_min[envs_idx] = self.cfg.goal_dist_min_initial
        self.keypoints_to_surface_dist_min[envs_idx] = 0.30
        self.near_goal_steps[envs_idx] = 0
        self.successes[envs_idx] = 0.0
        self.failed[envs_idx] = False
        self.coverage_counts[envs_idx] = 0.0
        self.episode_fingertip_contact_counts[envs_idx] = 0.0
        self.episode_fingertip_cluster_counts[envs_idx] = 0.0
        self.episode_contact_link_counts[envs_idx] = 0.0
        self.episode_contact_link_cluster_counts[envs_idx] = 0.0
        self.potential_per_kp_max[envs_idx] = 0.0
        self.contact_coverage_per_kp_max[envs_idx] = 0.0
        self.current_eef_quat[envs_idx] = self.default_eef_quat[envs_idx]
        self.last_wrench_force_norm[envs_idx] = 0.0
        self.last_wrench_torque_norm[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = False
        self.last_actions[envs_idx] = 0.0
        self.prev_actions[envs_idx] = 0.0

    def _compute_reward(self, actions: torch.Tensor):
        self.contact_info = self._compute_contact_info()
        target_pos = self.objects[0].get_pos()
        goal_dist = torch.norm(target_pos - self.goal_pos, dim=-1)
        contact_satisfied = self.contact_info["hand_contact"] & self.contact_info["env_detach"]
        progress = torch.clamp(self.goal_dist_min - goal_dist, min=0.0) * contact_satisfied.float()
        self.goal_dist_min = torch.where(contact_satisfied, torch.minimum(self.goal_dist_min, goal_dist), self.goal_dist_min)

        keypoint_surface_distances = self._keypoint_surface_distances()
        keypoint_delta = torch.clamp(self.keypoints_to_surface_dist_min - keypoint_surface_distances, min=0.0)
        self.keypoints_to_surface_dist_min = torch.minimum(
            self.keypoints_to_surface_dist_min,
            keypoint_surface_distances,
        )
        reach = keypoint_delta.mean(dim=1) * 20.0

        neighbor_pos = torch.stack([obj.get_pos() for obj in self.objects[1:]], dim=1)
        neighbor_move = torch.norm(neighbor_pos - self.initial_neighbor_pos, dim=-1).mean(dim=1)
        stability = -neighbor_move
        action_penalty = -torch.mean(actions.square(), dim=-1)
        non_fingertip_target_penalty = -self.contact_info["non_fingertip_hand_target_contact"].float()
        near_goal = (goal_dist < self.cfg.success_dist) & contact_satisfied
        self.near_goal_steps = (self.near_goal_steps + near_goal.to(gs.tc_int)) * near_goal.to(gs.tc_int)
        success = (self.near_goal_steps >= self.cfg.success_steps).float()
        self.successes = torch.maximum(self.successes, success)
        failed = self._compute_failure_mask(self.contact_info)
        self.failed = failed
        coverage_reward, coverage_info = self._compute_contact_coverage_reward(self.contact_info)
        reach_curiosity = coverage_info["reach_curiosity_rew"]

        terms = {
            "goal_dist": goal_dist.detach(),
            "target_progress": progress,
            "goal_position_dist_min": self.goal_dist_min.detach(),
            "reach": reach,
            "keypoint_surface_distances": keypoint_surface_distances.detach(),
            "keypoints_to_surface_dist_min": self.keypoints_to_surface_dist_min.detach(),
            "contact_coverage": coverage_reward,
            "reach_curiosity": reach_curiosity,
            "neighbor_stability": stability,
            "action_penalty": action_penalty,
            "non_fingertip_target_penalty": non_fingertip_target_penalty,
            "success": success,
            "near_goal_bonus": near_goal.float(),
            **coverage_info,
            **self.contact_info,
            "near_goal": near_goal.float(),
            "near_goal_steps": self.near_goal_steps.float(),
            "successes": self.successes,
            "failed": failed.float(),
            "wrench_force_norm": self.last_wrench_force_norm,
            "wrench_torque_norm": self.last_wrench_torque_norm,
        }
        reward = (
            self.reward_cfg["target_progress"] * progress
            + self.reward_cfg["reach"] * reach
            + self.reward_cfg["reach_curiosity"] * reach_curiosity
            + self.reward_cfg["contact_coverage"] * coverage_reward
            + self.reward_cfg.get("contact_diversity", 0.0) * coverage_info["contact_diversity_reward"]
            + self.reward_cfg["neighbor_stability"] * stability
            + self.reward_cfg["action_penalty"] * action_penalty
            + self.reward_cfg.get("non_fingertip_target_penalty", 0.0) * non_fingertip_target_penalty
            + self.reward_cfg["near_goal_bonus"] * near_goal.float()
            + self.reward_cfg["success"] * success
        )
        return reward, terms

    def _compute_control_targets(self, actions: torch.Tensor) -> torch.Tensor:
        full_target = self.current_targets.clone()
        if self.cfg.use_arm_ik and self.eef_link is not None and self.arm_dofs_idx:
            eef_pos = self._link_pos(self.eef_link)
            target_pos = eef_pos + actions[:, :3] * self.cfg.arm_pos_action_scale
            target_pos[:, 2] = target_pos[:, 2].clamp(self.cfg.table_pos[2] + 0.08, self.cfg.table_pos[2] + 0.45)
            target_pos[:, 0] = target_pos[:, 0].clamp(-0.30, 0.30)
            target_pos[:, 1] = target_pos[:, 1].clamp(-0.20, 0.55)
            delta_quat = self._quat_from_rotvec(actions[:, 3:6] * self.cfg.arm_rot_action_scale)
            self.current_eef_quat = self._quat_normalize(self._quat_mul(delta_quat, self.current_eef_quat))
            try:
                q_ik = self.robot.inverse_kinematics(
                    link=self.eef_link,
                    pos=target_pos,
                    quat=self.current_eef_quat,
                    dofs_idx_local=self.arm_dofs_idx,
                    rot_mask=[True, True, True],
                )
                full_target[:, self.arm_dofs_idx] = q_ik[:, self.arm_dofs_idx]
            except Exception:
                full_target[:, : len(self.arm_dofs_idx)] += actions[:, : len(self.arm_dofs_idx)] * self.cfg.arm_pos_action_scale
        else:
            full_target[:, : self.control_dim] += actions * self.cfg.action_scale

        if self.finger_dofs_idx:
            finger_actions = actions[:, 6 : 6 + len(self.finger_dofs_idx)]
            full_target[:, self.finger_dofs_idx] += finger_actions * self.cfg.finger_action_scale
        return full_target

    def _compute_contact_info(self) -> dict[str, torch.Tensor]:
        contacts = self.scene.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a = contacts["link_a"]
        if link_a.shape[-1] == 0:
            return self._empty_contact_info()
        link_b = contacts["link_b"]
        force = contacts["force"]
        position = contacts["position"]

        target_a = link_a == self.target_link_idx
        target_b = link_b == self.target_link_idx
        target_contact = target_a | target_b
        tip_a = torch.isin(link_a, self.fingertip_contact_link_indices)
        tip_b = torch.isin(link_b, self.fingertip_contact_link_indices)
        robot_a = torch.isin(link_a, self.robot_link_indices)
        robot_b = torch.isin(link_b, self.robot_link_indices)
        hand_a = torch.isin(link_a, self.hand_link_indices)
        hand_b = torch.isin(link_b, self.hand_link_indices)
        arm_a = torch.isin(link_a, self.arm_link_indices)
        arm_b = torch.isin(link_b, self.arm_link_indices)
        table_a = link_a == self.table_link_idx
        table_b = link_b == self.table_link_idx
        other_obj_a = torch.isin(link_a, self.neighbor_link_indices)
        other_obj_b = torch.isin(link_b, self.neighbor_link_indices)

        tip_target_contact = target_contact & (tip_a | tip_b)
        robot_target_contact = target_contact & (robot_a | robot_b)
        hand_target_contact = target_contact & (hand_a | hand_b)
        non_fingertip_hand_target_contact = target_contact & (hand_a | hand_b) & ~(tip_a | tip_b)
        table_contact = target_contact & (table_a | table_b)
        neighbor_contact = target_contact & (other_obj_a | other_obj_b)
        non_target_robot_contact = (robot_a | robot_b) & ~target_contact
        non_target_arm_contact = (arm_a | arm_b) & ~target_contact

        force_mag = torch.norm(force, dim=-1)
        active = force_mag > self.cfg.contact_force_threshold
        tip_target_contact &= active
        robot_target_contact &= active
        hand_target_contact &= active
        non_fingertip_hand_target_contact &= active
        table_contact &= active
        neighbor_contact &= active
        non_target_robot_contact &= active
        non_target_arm_contact &= active

        y_displacement = self.objects[0].get_pos()[:, 1] - self.initial_target_pos[:, 1]
        in_max_displacement = y_displacement.abs() < 0.08
        tip_contact = tip_target_contact.any(dim=-1)
        hand_target_contact_env = hand_target_contact.any(dim=-1)
        table_contact_env = table_contact.any(dim=-1)
        neighbor_contact_env = neighbor_contact.any(dim=-1)
        non_target_robot_contact_env = non_target_robot_contact.any(dim=-1)
        non_target_arm_contact_env = non_target_arm_contact.any(dim=-1)
        non_fingertip_hand_target_contact_env = non_fingertip_hand_target_contact.any(dim=-1)
        non_fingertip_hand_target_force_norm = torch.where(
            non_fingertip_hand_target_contact,
            force_mag,
            torch.zeros_like(force_mag),
        ).amax(dim=-1)
        non_target_arm_contact_force_norm = torch.where(
            non_target_arm_contact,
            force_mag,
            torch.zeros_like(force_mag),
        ).amax(dim=-1)
        env_detach = ~neighbor_contact_env
        if self.cfg.require_fingertip_target_contact:
            hand_contact = tip_contact
        else:
            hand_contact = hand_target_contact_env
        contact_count = tip_target_contact.sum(dim=-1).float()
        hand_contact_count = hand_target_contact.sum(dim=-1).float()

        contact_pos = torch.where(tip_target_contact.unsqueeze(-1), position, torch.zeros_like(position))
        keypoint_masks = []
        keypoint_positions = []
        keypoint_force_norms = []
        for group_indices in self.fingertip_contact_group_indices:
            kp_contact = tip_target_contact & (torch.isin(link_a, group_indices) | torch.isin(link_b, group_indices))
            keypoint_masks.append(kp_contact.any(dim=-1))
            kp_count = kp_contact.sum(dim=-1).clamp_min(1).to(gs.tc_float)
            keypoint_positions.append((position * kp_contact.unsqueeze(-1)).sum(dim=1) / kp_count.unsqueeze(-1))
            keypoint_force_norms.append(torch.where(kp_contact, force_mag, torch.zeros_like(force_mag)).amax(dim=-1))
        keypoint_contact_mask = torch.stack(keypoint_masks, dim=1)
        keypoint_contact_positions = torch.stack(keypoint_positions, dim=1)
        keypoint_contact_force_norm = torch.stack(keypoint_force_norms, dim=1)
        diversity_masks = []
        diversity_positions = []
        diversity_force_norms = []
        for link_idx in self.contact_diversity_link_indices:
            link_contact = hand_target_contact & ((link_a == link_idx) | (link_b == link_idx))
            diversity_masks.append(link_contact.any(dim=-1))
            link_count = link_contact.sum(dim=-1).clamp_min(1).to(gs.tc_float)
            diversity_positions.append((position * link_contact.unsqueeze(-1)).sum(dim=1) / link_count.unsqueeze(-1))
            diversity_force_norms.append(torch.where(link_contact, force_mag, torch.zeros_like(force_mag)).amax(dim=-1))
        if diversity_masks:
            contact_diversity_mask = torch.stack(diversity_masks, dim=1)
            contact_diversity_positions = torch.stack(diversity_positions, dim=1)
            contact_diversity_force_norm = torch.stack(diversity_force_norms, dim=1)
        else:
            contact_diversity_mask = torch.zeros((self.num_envs, 0), dtype=gs.tc_bool, device=self.device)
            contact_diversity_positions = torch.zeros((self.num_envs, 0, 3), dtype=gs.tc_float, device=self.device)
            contact_diversity_force_norm = torch.zeros((self.num_envs, 0), dtype=gs.tc_float, device=self.device)
        return {
            "hand_contact": hand_contact,
            "hand_target_contact": hand_target_contact_env,
            "in_max_displacement": in_max_displacement,
            "tip_contact": tip_contact,
            "opposing_normals_ok": torch.ones_like(tip_contact),
            "table_contact": table_contact_env,
            "neighbor_contact": neighbor_contact_env,
            "non_target_robot_contact": non_target_robot_contact_env,
            "non_target_arm_contact": non_target_arm_contact_env,
            "non_target_arm_contact_force_norm": non_target_arm_contact_force_norm,
            "non_fingertip_hand_target_contact": non_fingertip_hand_target_contact_env,
            "non_fingertip_hand_target_force_norm": non_fingertip_hand_target_force_norm,
            "env_detach": env_detach,
            "transition_contact": tip_contact & ~table_contact_env & env_detach,
            "contact_count": contact_count,
            "hand_contact_count": hand_contact_count,
            "contact_mask": tip_target_contact,
            "contact_positions": contact_pos,
            "contact_force_norm": torch.where(tip_target_contact, force_mag, torch.zeros_like(force_mag)).amax(dim=-1),
            "keypoint_contact_mask": keypoint_contact_mask,
            "keypoint_contact_positions": keypoint_contact_positions,
            "keypoint_contact_force_norm": keypoint_contact_force_norm,
            "contact_diversity_mask": contact_diversity_mask,
            "contact_diversity_positions": contact_diversity_positions,
            "contact_diversity_force_norm": contact_diversity_force_norm,
        }

    def _empty_contact_info(self) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        bools = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=self.device)
        return {
            "hand_contact": bools,
            "hand_target_contact": bools,
            "in_max_displacement": bools,
            "tip_contact": bools,
            "opposing_normals_ok": torch.ones_like(bools),
            "table_contact": bools,
            "neighbor_contact": bools,
            "non_target_robot_contact": bools,
            "non_target_arm_contact": bools,
            "non_target_arm_contact_force_norm": zeros,
            "non_fingertip_hand_target_contact": bools,
            "non_fingertip_hand_target_force_norm": zeros,
            "env_detach": torch.ones_like(bools),
            "transition_contact": bools,
            "contact_count": zeros,
            "hand_contact_count": zeros,
            "contact_mask": torch.zeros((self.num_envs, 0), dtype=gs.tc_bool, device=self.device),
            "contact_positions": torch.zeros((self.num_envs, 0, 3), dtype=gs.tc_float, device=self.device),
            "contact_force_norm": zeros,
            "keypoint_contact_mask": torch.zeros(
                (self.num_envs, len(LEAP_FINGERTIP_LINKS)), dtype=gs.tc_bool, device=self.device
            ),
            "keypoint_contact_positions": torch.zeros(
                (self.num_envs, len(LEAP_FINGERTIP_LINKS), 3), dtype=gs.tc_float, device=self.device
            ),
            "keypoint_contact_force_norm": torch.zeros(
                (self.num_envs, len(LEAP_FINGERTIP_LINKS)), dtype=gs.tc_float, device=self.device
            ),
            "contact_diversity_mask": torch.zeros(
                (self.num_envs, len(self.contact_diversity_links)), dtype=gs.tc_bool, device=self.device
            ),
            "contact_diversity_positions": torch.zeros(
                (self.num_envs, len(self.contact_diversity_links), 3), dtype=gs.tc_float, device=self.device
            ),
            "contact_diversity_force_norm": torch.zeros(
                (self.num_envs, len(self.contact_diversity_links)), dtype=gs.tc_float, device=self.device
            ),
        }

    def _compute_failure_mask(self, contact_info: dict[str, torch.Tensor]) -> torch.Tensor:
        failed = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=self.device)
        if self.cfg.fail_on_target_fall:
            min_z = self.cfg.table_pos[2] - self.cfg.target_fall_z_margin
            failed |= self.objects[0].get_pos()[:, 2] < min_z
        if self.cfg.fail_on_non_target_robot_contact:
            arm_force_threshold = float(getattr(self.cfg, "arm_contact_force_threshold", 1.0))
            failed |= contact_info["non_target_arm_contact_force_norm"] > arm_force_threshold
        return failed

    def _apply_domain_randomization(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.randomize_dynamics or env_ids.numel() == 0:
            return
        n = int(env_ids.numel())
        object_links = [0]
        for obj in self.objects:
            if self.cfg.randomize_friction:
                low, high = self.cfg.friction_ratio_range
                friction_ratio = low + (high - low) * torch.rand((n, 1), dtype=gs.tc_float, device=self.device)
                try:
                    obj.set_friction_ratio(friction_ratio=friction_ratio, links_idx_local=object_links, envs_idx=env_ids)
                except Exception:
                    pass
            if self.cfg.randomize_mass:
                low, high = self.cfg.mass_shift_range
                mass_shift = low + (high - low) * torch.rand((n, 1), dtype=gs.tc_float, device=self.device)
                try:
                    obj.set_mass_shift(mass_shift=mass_shift, links_idx_local=object_links, envs_idx=env_ids)
                except Exception:
                    pass
            if self.cfg.randomize_com:
                low, high = self.cfg.com_shift_range
                com_shift = low + (high - low) * torch.rand((n, 1, 3), dtype=gs.tc_float, device=self.device)
                try:
                    obj.set_COM_shift(com_shift=com_shift, links_idx_local=object_links, envs_idx=env_ids)
                except Exception:
                    pass

        if self.cfg.randomize_pd_gains:
            p_low, p_high = self.cfg.p_gain_range
            d_low, d_high = self.cfg.d_gain_range
            kp = p_low + (p_high - p_low) * torch.rand((n, self.robot_dof_dim), dtype=gs.tc_float, device=self.device)
            kv = d_low + (d_high - d_low) * torch.rand((n, self.robot_dof_dim), dtype=gs.tc_float, device=self.device)
            try:
                self.robot.set_dofs_kp(kp, envs_idx=env_ids)
                self.robot.set_dofs_kv(kv, envs_idx=env_ids)
            except Exception:
                pass

    def _maybe_apply_random_wrench(self) -> None:
        self.last_wrench_force_norm.zero_()
        self.last_wrench_torque_norm.zero_()
        if self.cfg.wrench_prob <= 0.0 or (self.cfg.force_scale <= 0.0 and self.cfg.torque_scale <= 0.0):
            return
        mask = torch.rand(self.num_envs, device=self.device) < self.cfg.wrench_prob
        if not mask.any():
            return
        env_ids = mask.nonzero(as_tuple=False).flatten()
        if self.cfg.force_scale > 0.0:
            force = (torch.rand((env_ids.numel(), 1, 3), dtype=gs.tc_float, device=self.device) * 2.0 - 1.0)
            force = force * self.cfg.force_scale
            try:
                self.scene.rigid_solver.apply_links_external_force(
                    force=force,
                    links_idx=[self.target_link_idx],
                    envs_idx=env_ids,
                    ref="link_com",
                )
                self.last_wrench_force_norm[env_ids] = torch.norm(force.squeeze(1), dim=-1)
            except Exception:
                pass
        if self.cfg.torque_scale > 0.0:
            torque = (torch.rand((env_ids.numel(), 1, 3), dtype=gs.tc_float, device=self.device) * 2.0 - 1.0)
            torque = torque * self.cfg.torque_scale
            try:
                self.scene.rigid_solver.apply_links_external_torque(
                    torque=torque,
                    links_idx=[self.target_link_idx],
                    envs_idx=env_ids,
                    ref="link_com",
                )
                self.last_wrench_torque_norm[env_ids] = torch.norm(torque.squeeze(1), dim=-1)
            except Exception:
                pass

    def _compute_contact_coverage_reward(self, contact_info: dict[str, torch.Tensor]):
        zeros = torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device)
        if not self.cfg.enable_contact_coverage:
            return zeros, {
                "reach_curiosity_rew": zeros,
                "avg_potential": zeros,
                "cluster_novelty_reward": zeros,
                "stateid_entropy": zeros,
            }

        target_pos = self.objects[0].get_pos()
        target_quat = self.objects[0].get_quat()
        state_ids, state_metrics = self._contact_state_ids(target_pos, target_quat)
        valid_env = contact_info["hand_contact"] & contact_info["env_detach"]
        tips = self._fingertip_positions()
        pc_world = self._surface_points_world(target_pos, target_quat)
        normals_world = self._surface_normals_world(target_quat)
        dists = torch.cdist(tips, pc_world)
        nearest_dist, nearest_surface_idx = dists.min(dim=-1)

        state_counts = self.state_contact_counts[state_ids]  # (N, L, K)
        point_clusters = self.surface_point_to_cluster.view(1, 1, -1).expand(
            self.num_envs, len(LEAP_FINGERTIP_LINKS), -1
        )
        counts_per_point = state_counts.gather(2, point_clusters)
        novelty_per_point = self._novelty_from_counts(counts_per_point)
        kernel = self._potential_kernel(dists)
        if self.cfg.mask_backface_points:
            to_keypoint = tips.unsqueeze(2) - pc_world.unsqueeze(1)
            cos_theta = (to_keypoint * normals_world.unsqueeze(1)).sum(dim=-1) / (
                torch.norm(to_keypoint, dim=-1).clamp_min(1e-8)
                * torch.norm(normals_world, dim=-1).unsqueeze(1).clamp_min(1e-8)
            )
            kernel = kernel * torch.clamp(cos_theta, min=0.0, max=1.0)
        if self.cfg.mask_palm_inward_points:
            palm_dirs = self._fingertip_palm_dirs().unsqueeze(2)
            cos_phi = (palm_dirs * normals_world.unsqueeze(1)).sum(dim=-1) / (
                torch.norm(palm_dirs, dim=-1).clamp_min(1e-8)
                * torch.norm(normals_world, dim=-1).unsqueeze(1).clamp_min(1e-8)
            )
            kernel = kernel * torch.clamp(-cos_phi, min=0.0, max=1.0)
        phi = (novelty_per_point * kernel).sum(dim=-1)

        env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.state_running_max_mode == "state":
            prev_phi = self.potential_per_kp_max[env_ids, state_ids]
            delta_phi = torch.clamp(phi - prev_phi, min=0.0)
            shaped = delta_phi.mean(dim=-1) if self.cfg.use_potential_shaping else phi.mean(dim=-1)
            potential_reward = torch.where(valid_env, shaped, zeros)
            if valid_env.any():
                self.potential_per_kp_max[env_ids[valid_env], state_ids[valid_env]] = torch.maximum(
                    prev_phi[valid_env], phi[valid_env]
                )
        else:
            prev_phi = self.potential_per_kp_max
            delta_phi = torch.clamp(phi - prev_phi, min=0.0)
            shaped = delta_phi.mean(dim=-1) if self.cfg.use_potential_shaping else phi.mean(dim=-1)
            potential_reward = torch.where(valid_env, shaped, zeros)
            if valid_env.any():
                self.potential_per_kp_max[valid_env] = torch.maximum(prev_phi[valid_env], phi[valid_env])

        keypoint_contact_mask = contact_info["keypoint_contact_mask"]
        contact_pos = contact_info["keypoint_contact_positions"]
        canonical_contact = self._world_to_target_local(contact_pos, target_pos, target_quat)
        contact_dist = torch.cdist(canonical_contact, self.canonical_surface_points.unsqueeze(0).expand(self.num_envs, -1, -1))
        contact_nearest_dist, contact_surface_idx = contact_dist.min(dim=-1)
        contact_cluster = self.surface_point_to_cluster[contact_surface_idx]
        valid_contact = (
            keypoint_contact_mask
            & valid_env.unsqueeze(1)
            & (contact_nearest_dist < self.cfg.near_surface_threshold * 2.0)
        )
        diversity_contact_mask = contact_info["contact_diversity_mask"] & valid_env.unsqueeze(1)
        diversity_contact_pos = contact_info["contact_diversity_positions"]
        if diversity_contact_mask.shape[1] > 0:
            canonical_diversity_contact = self._world_to_target_local(
                diversity_contact_pos,
                target_pos,
                target_quat,
            )
            diversity_contact_dist = torch.cdist(
                canonical_diversity_contact,
                self.canonical_surface_points.unsqueeze(0).expand(self.num_envs, -1, -1),
            )
            diversity_contact_nearest_dist, diversity_contact_surface_idx = diversity_contact_dist.min(dim=-1)
            diversity_contact_cluster = self.surface_point_to_cluster[diversity_contact_surface_idx]
            diversity_contact_mask = diversity_contact_mask & (
                diversity_contact_nearest_dist < self.cfg.near_surface_threshold * 2.0
            )
            link_ids = torch.arange(diversity_contact_mask.shape[1], device=self.device).view(1, -1).expand_as(
                diversity_contact_cluster
            )
            diversity_prior_counts = self.episode_contact_link_cluster_counts[
                env_ids[:, None],
                link_ids,
                diversity_contact_cluster,
            ]
            new_link_cluster_contact = diversity_contact_mask & (diversity_prior_counts <= 0.0)
            new_link_cluster_contact_reward = torch.where(
                valid_env,
                new_link_cluster_contact.float().mean(dim=-1),
                zeros,
            )
        else:
            diversity_contact_cluster = torch.zeros((self.num_envs, 0), dtype=torch.long, device=self.device)
            new_link_cluster_contact_reward = zeros
        kp_ids = torch.arange(len(LEAP_FINGERTIP_LINKS), device=self.device).view(1, -1).expand_as(contact_cluster)
        episode_prior_counts = self.episode_fingertip_cluster_counts[
            env_ids[:, None],
            kp_ids,
            contact_cluster,
        ]
        new_episode_contact = valid_contact & (episode_prior_counts <= 0.0)
        new_episode_contact_reward = torch.where(
            valid_env,
            new_episode_contact.float().mean(dim=-1),
            zeros,
        )

        contact_novelty = torch.zeros_like(phi)
        if valid_contact.any():
            prior_counts = self.state_contact_counts[state_ids[:, None], kp_ids, contact_cluster]
            contact_novelty[valid_contact] = self._novelty_from_counts(prior_counts)[valid_contact]

        if self.cfg.state_running_max_mode == "state":
            prev_bonus = self.contact_coverage_per_kp_max[env_ids, state_ids]
            delta_bonus = torch.clamp(contact_novelty - prev_bonus, min=0.0)
            novelty_reward = torch.where(valid_env, delta_bonus.mean(dim=-1), zeros)
            if valid_env.any():
                self.contact_coverage_per_kp_max[env_ids[valid_env], state_ids[valid_env]] = torch.maximum(
                    prev_bonus[valid_env], contact_novelty[valid_env]
                )
        else:
            prev_bonus = self.contact_coverage_per_kp_max
            delta_bonus = torch.clamp(contact_novelty - prev_bonus, min=0.0)
            novelty_reward = torch.where(valid_env, delta_bonus.mean(dim=-1), zeros)
            if valid_env.any():
                self.contact_coverage_per_kp_max[valid_env] = torch.maximum(
                    prev_bonus[valid_env], contact_novelty[valid_env]
                )

        if valid_contact.any():
            vc_env, vc_kp = torch.nonzero(valid_contact, as_tuple=True)
            vc_cluster = contact_cluster[vc_env, vc_kp]
            ones = torch.ones_like(vc_cluster, dtype=gs.tc_float)
            self.episode_fingertip_contact_counts[vc_env, vc_kp] += 1.0
            self.episode_fingertip_cluster_counts.index_put_((vc_env, vc_kp, vc_cluster), ones, accumulate=True)
            if self.hash_state_bank is not None:
                self.hash_state_bank.add_contacts(
                    state_ids=state_ids,
                    contact_mask=valid_contact,
                    contact_bins=contact_cluster,
                )
                self.state_contact_counts = self.hash_state_bank.counts
            else:
                vc_state = state_ids[vc_env]
                self.state_contact_counts.index_put_((vc_state, vc_kp, vc_cluster), ones, accumulate=True)
            self.coverage_counts[vc_env, contact_surface_idx[vc_env, vc_kp]] += 1.0
            self.global_coverage_counts.scatter_add_(0, contact_surface_idx[vc_env, vc_kp], ones)
        if diversity_contact_mask.any():
            dc_env, dc_link = torch.nonzero(diversity_contact_mask, as_tuple=True)
            dc_cluster = diversity_contact_cluster[dc_env, dc_link]
            ones = torch.ones_like(dc_cluster, dtype=gs.tc_float)
            self.episode_contact_link_counts[dc_env, dc_link] += 1.0
            self.episode_contact_link_cluster_counts.index_put_((dc_env, dc_link, dc_cluster), ones, accumulate=True)

        occupied = (self.state_contact_counts[state_ids].sum(dim=1) > 0).float().mean(dim=-1)
        entropy = self._coverage_entropy(self.state_contact_counts[state_ids].sum(dim=1))
        finger_contact_entropy = self._coverage_entropy(self.episode_fingertip_contact_counts)
        current_finger_contact_entropy = self._coverage_entropy(valid_contact.float())
        link_contact_entropy = self._coverage_entropy(self.episode_contact_link_counts)
        current_link_contact_entropy = self._coverage_entropy(diversity_contact_mask.float())
        link_contact_count = diversity_contact_mask.sum(dim=1).float()
        contact_diversity_reward = new_episode_contact_reward + 0.25 * finger_contact_entropy
        return novelty_reward, {
            "reach_curiosity_rew": potential_reward,
            "avg_potential": phi.mean(dim=-1),
            "contact_count": valid_contact.sum(dim=1).float(),
            "link_contact_count": link_contact_count,
            "cluster_novelty_reward": novelty_reward,
            "new_episode_contact_reward": new_episode_contact_reward,
            "new_finger_cluster_contact_reward": new_episode_contact_reward,
            "new_link_cluster_contact_reward": new_link_cluster_contact_reward,
            "finger_contact_entropy": finger_contact_entropy,
            "current_finger_contact_entropy": current_finger_contact_entropy,
            "link_contact_entropy": link_contact_entropy,
            "current_link_contact_entropy": current_link_contact_entropy,
            "contact_diversity_reward": contact_diversity_reward,
            "stateid_entropy": entropy,
            "state_coverage": occupied,
            "cur_keypoints_to_surface_dist_min": nearest_dist.min(dim=-1).values,
            **state_metrics,
        }

    def _coverage_entropy(self, counts: torch.Tensor) -> torch.Tensor:
        probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=-1)
        return entropy / math.log(float(counts.shape[-1]))

    def _make_cube_surface_points(self) -> torch.Tensor:
        half = self.cfg.box_size[0] * 0.5
        grid = torch.linspace(-half, half, self.cfg.coverage_grid, dtype=gs.tc_float, device=self.device)
        yy, zz = torch.meshgrid(grid, grid, indexing="ij")
        faces = []
        for sign in (-1.0, 1.0):
            faces.append(torch.stack([torch.full_like(yy, sign * half), yy, zz], dim=-1).reshape(-1, 3))
            faces.append(torch.stack([yy, torch.full_like(yy, sign * half), zz], dim=-1).reshape(-1, 3))
            faces.append(torch.stack([yy, zz, torch.full_like(yy, sign * half)], dim=-1).reshape(-1, 3))
        return torch.cat(faces, dim=0)

    def _make_cube_surface_normals(self) -> torch.Tensor:
        grid_n = self.cfg.coverage_grid * self.cfg.coverage_grid
        normals = []
        for sign in (-1.0, 1.0):
            normals.append(torch.tensor([sign, 0.0, 0.0], dtype=gs.tc_float, device=self.device).repeat(grid_n, 1))
            normals.append(torch.tensor([0.0, sign, 0.0], dtype=gs.tc_float, device=self.device).repeat(grid_n, 1))
            normals.append(torch.tensor([0.0, 0.0, sign], dtype=gs.tc_float, device=self.device).repeat(grid_n, 1))
        return torch.cat(normals, dim=0)

    def _assign_surface_clusters(self) -> torch.Tensor:
        points = self.canonical_surface_points
        k = min(self.cfg.surface_cluster_k, points.shape[0])
        center_idx = self._fps_indices(points, k, start_idx=0)
        centers = points[center_idx]
        center_normals = self.canonical_surface_normals[center_idx]
        labels = torch.zeros(points.shape[0], dtype=torch.long, device=self.device)
        for _ in range(max(1, int(self.cfg.max_clustering_iters))):
            pos_dist = torch.cdist(points, centers)
            if self.cfg.use_normal_in_clustering:
                normal_sim = self.canonical_surface_normals @ center_normals.t()
                normal_dist = 1.0 - normal_sim.clamp(-1.0, 1.0)
                dist = (1.0 - self.cfg.normal_weight) * pos_dist + self.cfg.normal_weight * normal_dist
            else:
                dist = pos_dist
            labels = dist.argmin(dim=-1)
            for cluster_id in range(k):
                mask = labels == cluster_id
                if mask.any():
                    centers[cluster_id] = points[mask].mean(dim=0)
                    if self.cfg.use_normal_in_clustering:
                        normal = self.canonical_surface_normals[mask].mean(dim=0)
                        center_normals[cluster_id] = normal / torch.norm(normal).clamp_min(1e-8)
        if k < self.cfg.surface_cluster_k:
            labels = labels.clamp_max(k - 1)
        return labels.to(torch.long)

    def _make_state_point_indices(self) -> torch.Tensor:
        n_points = self.canonical_surface_points.shape[0]
        n_take = min(max(1, int(self.cfg.state_num_points)), n_points)
        return self._fps_indices(self.canonical_surface_points, n_take, start_idx=0)

    def _fps_indices(self, points: torch.Tensor, k: int, start_idx: int = 0) -> torch.Tensor:
        points = points.to(torch.float32)
        n_points = int(points.shape[0])
        k = min(int(k), n_points)
        dists = torch.full((n_points,), float("inf"), dtype=gs.tc_float, device=self.device)
        indices = torch.empty((k,), dtype=torch.long, device=self.device)
        farthest = int(start_idx) % n_points
        for i in range(k):
            indices[i] = farthest
            dist2 = (points - points[farthest].view(1, -1)).square().sum(dim=1)
            dists = torch.minimum(dists, dist2)
            farthest = int(torch.argmax(dists).item())
        return indices

    def _make_hash_state_bank(self) -> LearnedHashStateBank | None:
        if self.cfg.state_type == "predefined":
            return None
        if self.cfg.state_type != "hash":
            raise ValueError(f"Unsupported state_type: {self.cfg.state_type}")
        feature_dim = int(self.state_point_indices.numel()) * 3
        if self.cfg.state_include_goal:
            feature_dim *= 2
        return LearnedHashStateBank(
            num_key_states=self.cfg.num_key_states,
            feature_dim=feature_dim,
            buffer_size=max(self.num_envs * int(self.cfg.hash_ae_update_freq), 64),
            num_hand_keypoints=len(LEAP_FINGERTIP_LINKS),
            num_object_bins=self.cfg.surface_cluster_k,
            device=self.device,
            code_dim=self.cfg.hash_code_dim,
            hidden_dim=self.cfg.hash_hidden_dim,
            noise_scale=self.cfg.hash_noise_scale,
            lambda_binary=self.cfg.hash_lambda_binary,
            ae_lr=self.cfg.hash_ae_lr,
            ae_update_steps=self.cfg.hash_ae_steps,
            ae_update_freq=self.cfg.hash_ae_update_freq,
            ae_num_minibatches=self.cfg.hash_ae_num_minibatches,
            seed=self.cfg.hash_seed,
        )

    def _infer_curiosity_state_dim(self) -> int:
        if self.cfg.curiosity_state_type == "policy_state":
            return self.robot_q_dim + self.robot_q_dim + 3 + 3 + 4 + 3 + 12 + 1 + 4 + self.num_actions
        if self.cfg.curiosity_state_type == "contact_force":
            return len(LEAP_FINGERTIP_LINKS)
        if self.cfg.curiosity_state_type == "contact_distance":
            return len(LEAP_FINGERTIP_LINKS) * 3
        if self.cfg.curiosity_state_type == "state_feature":
            dim = int(self.state_point_indices.numel()) * 3
            return dim * (2 if self.cfg.state_include_goal else 1)
        raise ValueError(f"Unsupported curiosity_state_type: {self.cfg.curiosity_state_type}")

    def _current_curiosity_states(self) -> torch.Tensor:
        if self.cfg.curiosity_state_type == "policy_state":
            return self.get_observations()["policy"]
        if self.cfg.curiosity_state_type == "contact_force":
            return self.contact_info["keypoint_contact_force_norm"].reshape(self.num_envs, -1)
        if self.cfg.curiosity_state_type == "contact_distance":
            target_pos = self.objects[0].get_pos()
            return (self._fingertip_positions() - target_pos[:, None, :]).reshape(self.num_envs, -1)
        if self.cfg.curiosity_state_type == "state_feature":
            return self._build_state_features(self.objects[0].get_pos(), self.objects[0].get_quat())
        raise ValueError(f"Unsupported curiosity_state_type: {self.cfg.curiosity_state_type}")

    def _contact_state_ids(
        self, target_pos: torch.Tensor, target_quat: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.hash_state_bank is None:
            return self._predefined_state_ids(target_pos), {}
        state_features = self._build_state_features(target_pos, target_quat)
        self.hash_state_bank.push(state_features)
        state_ids = self.hash_state_bank.assign(state_features)
        metrics = {
            key: value.repeat(self.num_envs) if value.ndim == 0 else value
            for key, value in self.hash_state_bank.get_metrics().items()
        }
        return state_ids, metrics

    def _build_state_features(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        target_pc_all = self._surface_points_world(target_pos, target_quat)
        target_pc = target_pc_all.index_select(1, self.state_point_indices)
        features = [target_pc.reshape(self.num_envs, -1)]
        if self.cfg.state_include_goal:
            goal_pc = self.canonical_surface_points[self.state_point_indices].unsqueeze(0) + self.goal_pos[:, None, :]
            features.append(goal_pc.reshape(self.num_envs, -1))
        return torch.cat(features, dim=-1).to(torch.float32)

    def _predefined_state_ids(self, target_pos: torch.Tensor) -> torch.Tensor:
        start = self.initial_target_pos
        goal = self.goal_pos
        total = torch.norm(goal - start, dim=-1).clamp_min(1e-6)
        progress = torch.sum((target_pos - start) * (goal - start), dim=-1) / total.square()
        progress = progress.clamp(0.0, 0.999)
        return torch.floor(progress * self.cfg.num_key_states).to(torch.long).clamp(0, self.cfg.num_key_states - 1)

    def _surface_points_world(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        points = self.canonical_surface_points.unsqueeze(0).expand(self.num_envs, -1, -1)
        quat = target_quat.unsqueeze(1).expand(-1, points.shape[1], -1)
        return self._quat_apply(quat.reshape(-1, 4), points.reshape(-1, 3)).view_as(points) + target_pos[:, None, :]

    def _surface_normals_world(self, target_quat: torch.Tensor) -> torch.Tensor:
        normals = self.canonical_surface_normals.unsqueeze(0).expand(self.num_envs, -1, -1)
        quat = target_quat.unsqueeze(1).expand(-1, normals.shape[1], -1)
        return self._quat_apply(quat.reshape(-1, 4), normals.reshape(-1, 3)).view_as(normals)

    def _world_to_target_local(
        self, points_world: torch.Tensor, target_pos: torch.Tensor, target_quat: torch.Tensor
    ) -> torch.Tensor:
        rel = points_world - target_pos[:, None, :]
        quat_inv = self._quat_conjugate(target_quat).unsqueeze(1).expand(-1, rel.shape[1], -1)
        return self._quat_apply(quat_inv.reshape(-1, 4), rel.reshape(-1, 3)).view_as(rel)

    def _potential_kernel(self, distances: torch.Tensor) -> torch.Tensor:
        if self.cfg.potential_kernel == "inverse":
            return 1.0 / (distances + self.cfg.curiosity_kernel_param)
        if self.cfg.potential_kernel == "gaussian":
            return torch.exp(-distances.square() / (2.0 * self.cfg.curiosity_kernel_param**2))
        if self.cfg.potential_kernel == "exponential":
            return torch.exp(-distances / self.cfg.curiosity_kernel_param)
        raise ValueError(f"Unsupported potential_kernel: {self.cfg.potential_kernel}")

    def _novelty_from_counts(self, counts: torch.Tensor) -> torch.Tensor:
        if self.cfg.novelty_decay == "exponential":
            return torch.exp(-self.cfg.novelty_decay_rate * counts)
        if self.cfg.novelty_decay == "linear":
            return 1.0 / (1.0 + self.cfg.novelty_decay_rate * counts)
        if self.cfg.novelty_decay == "sqrt":
            return 1.0 / torch.sqrt(1.0 + self.cfg.novelty_decay_rate * counts)
        if self.cfg.novelty_decay == "logarithmic":
            return 1.0 / torch.log(2.0 + counts)
        raise ValueError(f"Unsupported novelty_decay: {self.cfg.novelty_decay}")

    def _keypoint_surface_distances(self) -> torch.Tensor:
        target_pos = self.objects[0].get_pos()
        target_quat = self.objects[0].get_quat()
        tips = self._fingertip_positions()
        pc_world = self._surface_points_world(target_pos, target_quat)
        return torch.cdist(tips, pc_world).min(dim=-1).values

    def _get_link_or_none(self, name: str):
        try:
            return self.robot.get_link(name)
        except Exception:
            return None

    def _link_pos(self, link) -> torch.Tensor:
        if link is None:
            return torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=self.device)
        return link.get_pos()

    def _link_quat(self, link) -> torch.Tensor:
        if link is None:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=self.device).repeat(self.num_envs, 1)
        return link.get_quat()

    def _quat_from_rotvec(self, rotvec: torch.Tensor) -> torch.Tensor:
        angle = torch.norm(rotvec, dim=-1, keepdim=True)
        half = 0.5 * angle
        axis = rotvec / angle.clamp_min(1e-8)
        sin_half = torch.sin(half)
        quat = torch.cat([torch.cos(half), axis * sin_half], dim=-1)
        small = angle.squeeze(-1) < 1e-8
        if small.any():
            quat[small] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=self.device)
        return quat

    def _quat_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(dim=-1)
        bw, bx, by, bz = b.unbind(dim=-1)
        return torch.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dim=-1,
        )

    def _quat_conjugate(self, quat: torch.Tensor) -> torch.Tensor:
        out = quat.clone()
        out[..., 1:] = -out[..., 1:]
        return out

    def _quat_apply(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        quat = self._quat_normalize(quat)
        q_vec = quat[..., 1:]
        q_w = quat[..., :1]
        t = 2.0 * torch.cross(q_vec, vec, dim=-1)
        return vec + q_w * t + torch.cross(q_vec, t, dim=-1)

    def _quat_normalize(self, quat: torch.Tensor) -> torch.Tensor:
        return quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)

    def _fingertip_positions(self) -> torch.Tensor:
        if not self.fingertip_links:
            return self._link_pos(self.palm_link).unsqueeze(1).repeat(1, 4, 1)
        pos = [link.get_pos() for link in self.fingertip_links]
        while len(pos) < 4:
            pos.append(pos[-1])
        return torch.stack(pos[:4], dim=1)

    def _fingertip_palm_dirs(self) -> torch.Tensor:
        if not self.fingertip_links:
            palm = self._link_pos(self.palm_link).unsqueeze(1)
            tips = self._fingertip_positions()
            return (tips - palm) / torch.norm(tips - palm, dim=-1, keepdim=True).clamp_min(1e-8)
        axis = torch.tensor([-1.0, 0.0, 0.0], dtype=gs.tc_float, device=self.device)
        dirs = []
        for link in self.fingertip_links:
            quat = self._link_quat(link)
            dirs.append(self._quat_apply(quat, axis.unsqueeze(0).expand(self.num_envs, -1)))
        while len(dirs) < 4:
            dirs.append(dirs[-1])
        return torch.stack(dirs[:4], dim=1)

    def _safe_tensor(self, fn, shape: Iterable[int]) -> torch.Tensor:
        if fn is None:
            return torch.zeros(tuple(shape), dtype=gs.tc_float, device=self.device)
        try:
            value = fn()
            if value.shape[-1] < shape[-1]:
                pad = torch.zeros((self.num_envs, shape[-1] - value.shape[-1]), dtype=value.dtype, device=value.device)
                value = torch.cat([value, pad], dim=-1)
            return value[:, : shape[-1]]
        except Exception:
            return torch.zeros(tuple(shape), dtype=gs.tc_float, device=self.device)

    def _episode_log_terms(self, terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logs = {}
        skip = {"contact_mask", "contact_positions", "keypoint_contact_positions", "contact_diversity_positions"}
        for key, value in terms.items():
            if key in skip or not torch.is_tensor(value):
                continue
            tensor = value.detach()
            if tensor.dtype == torch.bool:
                tensor = tensor.float()
            if tensor.ndim > 1:
                tensor = tensor.reshape(self.num_envs, -1).mean(dim=-1)
            logs[f"rew_{key}"] = tensor.mean()
        return logs


def default_reward_cfg() -> dict[str, float]:
    return {
        "target_progress": 60.0,
        "reach": 0.5,
        "reach_curiosity": 1.28,
        "contact_coverage": 200.0,
        "contact_diversity": 0.0,
        "non_fingertip_target_penalty": 0.0,
        "neighbor_stability": 5.0,
        "action_penalty": 0.005,
        "near_goal_bonus": 10.0,
        "success": 4000.0,
    }


def default_train_cfg(exp_name: str) -> dict:
    return {
        "algorithm": {
            "class_name": "contactexplorer_genesis.curiosity_ppo.CuriosityPPO",
            "clip_param": 0.1,
            "desired_kl": 0.016,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "curiosity_cfg": {
                "enabled": False,
                "model_type": "prediction_error",
                "intrinsic_reward_scale": 1.0,
                "emb_dim": 8,
                "hidden_dims": [512, 256, 128],
                "activation": "elu",
                "ensemble_size": 5,
                "simhash_dim": 5,
                "code_dim": 16,
                "hash_hidden_dim": 512,
                "hash_noise_scale": 0.3,
                "hash_lambda_binary": 1.0,
                "obs_act_normalization": True,
                "curiosity_normalization": True,
                "learning_rate": 1e-4,
                "reward_scale": 0.01,
            },
            "normalize_value": True,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 2,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 12,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }
