from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from tensordict import TensorDict

from contactexplorer_genesis.curiosity_model import CuriosityModel


class ValueRunningMeanStd(nn.Module):
    def __init__(self, shape: tuple[int, ...] = (1,), eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update(self, x: torch.Tensor) -> None:
        if not self.training or x.shape[0] <= 1:
            return
        x64 = x.detach().to(torch.float64)
        batch_mean = x64.mean(dim=0)
        batch_var = x64.var(dim=0, unbiased=False)
        batch_count = torch.as_tensor(x64.shape[0], dtype=torch.float64, device=x.device)
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        new_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(m2 / total_count)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if update:
            self._update(flat)
        mean = self.running_mean.to(dtype=x.dtype, device=x.device)
        var = self.running_var.to(dtype=x.dtype, device=x.device)
        normalized = (flat - mean) / torch.sqrt(var + self.eps)
        return torch.clamp(normalized, -5.0, 5.0).reshape(shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.to(dtype=x.dtype, device=x.device)
        var = self.running_var.to(dtype=x.dtype, device=x.device)
        clipped = torch.clamp(x, -5.0, 5.0)
        return clipped * torch.sqrt(var + self.eps) + mean


class CuriosityRolloutStorage(RolloutStorage):
    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        curiosity_shape: tuple[int, ...] | list[int] | None,
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.curiosity_states = None
        if curiosity_shape is not None:
            self.curiosity_states = torch.zeros(num_transitions_per_env, num_envs, *curiosity_shape, device=device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        step = self.step
        super().add_transition(transition)
        if self.curiosity_states is not None:
            self.curiosity_states[step].copy_(transition.curiosity_states)  # type: ignore[attr-defined]

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore
        curiosity_states = self.curiosity_states.flatten(0, 1) if self.curiosity_states is not None else None

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )
                if curiosity_states is not None:
                    batch.curiosity_states = curiosity_states[batch_idx]
                yield batch


class CuriosityPPO(PPO):
    def __init__(
        self,
        *args,
        curiosity_cfg: dict | None = None,
        curiosity_state_dim: int | None = None,
        normalize_value: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.curiosity_cfg = curiosity_cfg or {}
        self.use_curiosity_model = bool(self.curiosity_cfg.get("enabled", False))
        self.normalize_value = normalize_value
        self.value_running_mean_std = ValueRunningMeanStd((1,)).to(self.device) if self.normalize_value else None
        self.intrinsic_reward_scale = float(self.curiosity_cfg.get("intrinsic_reward_scale", 1.0))
        self.reward_scale = float(self.curiosity_cfg.get("reward_scale", 1.0))
        self.intrinsic_rewards = None
        self.curiosity_model = None
        self.curiosity_optimizer = None
        self.mean_intrinsic_reward = 0.0

        if self.use_curiosity_model:
            if curiosity_state_dim is None:
                raise ValueError("curiosity_state_dim is required when PPO-side curiosity is enabled")
            policy_obs_dim = int(self.storage.observations["policy"].shape[-1])
            action_dim = int(self.storage.actions.shape[-1])
            self.curiosity_model = CuriosityModel(
                model_type=self.curiosity_cfg.get("model_type", "prediction_error"),
                obs_dim=policy_obs_dim,
                action_dim=action_dim,
                curiosity_dim=int(curiosity_state_dim),
                emb_dim=int(self.curiosity_cfg.get("emb_dim", 32)),
                hidden_dims=list(self.curiosity_cfg.get("hidden_dims", [256, 256])),
                activation=self.curiosity_cfg.get("activation", "elu"),
                ensemble_size=int(self.curiosity_cfg.get("ensemble_size", 5)),
                simhash_dim=int(self.curiosity_cfg.get("simhash_dim", 5)),
                code_dim=int(self.curiosity_cfg.get("code_dim", 16)),
                hash_hidden_dim=int(self.curiosity_cfg.get("hash_hidden_dim", 512)),
                hash_noise_scale=float(self.curiosity_cfg.get("hash_noise_scale", 0.3)),
                hash_lambda_binary=float(self.curiosity_cfg.get("hash_lambda_binary", 1.0)),
                obs_act_normalization=bool(self.curiosity_cfg.get("obs_act_normalization", True)),
                curiosity_normalization=bool(self.curiosity_cfg.get("curiosity_normalization", True)),
                device=self.device,
            )
            self.curiosity_optimizer = resolve_optimizer(self.curiosity_cfg.get("optimizer", "adam"))(
                self.curiosity_model.parameters_module.parameters(),
                lr=float(self.curiosity_cfg.get("learning_rate", 1e-4)),
                eps=float(self.curiosity_cfg.get("eps", 1e-5)),
            )

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        if self.value_running_mean_std is not None:
            self.transition.values = self.value_running_mean_std.inverse(self.transition.values)
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone() * self.reward_scale
        self.transition.dones = dones

        if self.use_curiosity_model:
            next_curiosity = extras["curiosity_states"].to(self.device).reshape(rewards.shape[0], -1)
            policy_obs = self.transition.observations["policy"].to(self.device)  # type: ignore[index]
            actions = self.transition.actions.to(self.device)  # type: ignore[union-attr]
            self.curiosity_model.update_normalization(policy_obs, actions, next_curiosity)  # type: ignore[union-attr]
            intrinsic = self.curiosity_model.compute_intrinsic_reward(policy_obs, actions, next_curiosity)  # type: ignore[union-attr]
            intrinsic = self.intrinsic_reward_scale * intrinsic.detach()
            self.intrinsic_rewards = intrinsic
            self.mean_intrinsic_reward = float(intrinsic.mean().item())
            self.transition.rewards += intrinsic
            self.transition.curiosity_states = next_curiosity.detach()
            extras.setdefault("episode", {})
            extras["episode"]["rew_ppo_intrinsic_reward"] = intrinsic.mean()
        elif self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1  # type: ignore[operator]
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        if self.value_running_mean_std is not None:
            last_values = self.value_running_mean_std.inverse(last_values)
        self.critic.reset(hidden_state=critic_hidden_state)

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        if self.value_running_mean_std is not None:
            st.values = self.value_running_mean_std.normalize(st.values)
            st.returns = self.value_running_mean_std.normalize(st.returns)

        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_curiosity_loss = 0 if self.use_curiosity_model else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)  # type: ignore
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            curiosity_loss = None
            if self.use_curiosity_model:
                policy_obs = batch.observations["policy"][:original_batch_size]
                action = batch.actions[:original_batch_size]
                curiosity = batch.curiosity_states[:original_batch_size]
                curiosity_loss = self.curiosity_model.compute_loss(policy_obs, action, curiosity)  # type: ignore[union-attr]
                self.curiosity_optimizer.zero_grad()  # type: ignore[union-attr]
                curiosity_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()
            if self.use_curiosity_model:
                nn.utils.clip_grad_norm_(self.curiosity_model.parameters_module.parameters(), self.max_grad_norm)  # type: ignore[union-attr]
                self.curiosity_optimizer.step()  # type: ignore[union-attr]

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if mean_curiosity_loss is not None:
                mean_curiosity_loss += curiosity_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        if mean_curiosity_loss is not None:
            loss_dict["curiosity"] = mean_curiosity_loss / num_updates
            loss_dict["ppo_intrinsic_reward"] = self.mean_intrinsic_reward

        self.storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        if self.value_running_mean_std is not None:
            self.value_running_mean_std.train()
        if self.curiosity_model is not None:
            self.curiosity_model.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.value_running_mean_std is not None:
            self.value_running_mean_std.eval()
        if self.curiosity_model is not None:
            self.curiosity_model.eval()

    def save(self) -> dict:
        saved = super().save()
        if self.value_running_mean_std is not None:
            saved["value_running_mean_std_state_dict"] = self.value_running_mean_std.state_dict()
        if self.use_curiosity_model:
            saved["curiosity_model_state_dict"] = self.curiosity_model.state_dict()  # type: ignore[union-attr]
            saved["curiosity_optimizer_state_dict"] = self.curiosity_optimizer.state_dict()  # type: ignore[union-attr]
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if self.value_running_mean_std is not None and "value_running_mean_std_state_dict" in loaded_dict:
            self.value_running_mean_std.load_state_dict(
                loaded_dict["value_running_mean_std_state_dict"],
                strict=strict,
            )
        if self.use_curiosity_model and "curiosity_model_state_dict" in loaded_dict:
            self.curiosity_model.load_state_dict(loaded_dict["curiosity_model_state_dict"], strict=strict)  # type: ignore[union-attr]
            self.curiosity_optimizer.load_state_dict(loaded_dict["curiosity_optimizer_state_dict"])  # type: ignore[union-attr]
        return load_iteration

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        if self.use_curiosity_model:
            model_params = [self.curiosity_model.model.state_dict()]  # type: ignore[union-attr]
            torch.distributed.broadcast_object_list(model_params, src=0)
            self.curiosity_model.model.load_state_dict(model_params[0])  # type: ignore[union-attr]

    def reduce_parameters(self) -> None:
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        if self.use_curiosity_model:
            all_params = chain(all_params, self.curiosity_model.parameters_module.parameters())  # type: ignore[union-attr]
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        curiosity_cfg = cfg["algorithm"].pop("curiosity_cfg", None)
        curiosity_enabled = bool(curiosity_cfg and curiosity_cfg.get("enabled", False))
        curiosity_state_dim = getattr(env, "curiosity_state_dim", None) if curiosity_enabled else None
        curiosity_shape = [int(curiosity_state_dim)] if curiosity_enabled else None
        storage = CuriosityRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            curiosity_shape,
            device,
        )

        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            curiosity_cfg=curiosity_cfg,
            curiosity_state_dim=curiosity_state_dim,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
