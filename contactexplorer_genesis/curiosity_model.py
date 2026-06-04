from __future__ import annotations

import torch
from torch import nn

from contactexplorer_genesis.hash_state_bank import HashAutoEncoder, RunningNormalizer, bits_to_int


def make_mlp(in_dim: int, out_dim: int, hidden_dims: list[int], activation: str = "elu") -> nn.Sequential:
    act_cls = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }.get(activation, nn.ELU)
    layers: list[nn.Module] = []
    last = int(in_dim)
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, int(hidden)))
        layers.append(act_cls())
        last = int(hidden)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class CuriosityModel:
    """ContactExplorer-style PPO-side intrinsic curiosity module."""

    def __init__(
        self,
        *,
        model_type: str,
        obs_dim: int,
        action_dim: int,
        curiosity_dim: int,
        emb_dim: int = 32,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
        ensemble_size: int = 5,
        simhash_dim: int = 5,
        code_dim: int = 16,
        hash_hidden_dim: int = 512,
        hash_noise_scale: float = 0.3,
        hash_lambda_binary: float = 1.0,
        obs_act_normalization: bool = True,
        curiosity_normalization: bool = True,
        device: torch.device | str = "cpu",
    ):
        self.model_type = str(model_type)
        self.device = torch.device(device)
        self.hash_noise_scale = float(hash_noise_scale)
        self.hash_lambda_binary = float(hash_lambda_binary)
        hidden_dims = hidden_dims or [256, 256]

        if self.model_type == "prediction_error":
            self.model = make_mlp(obs_dim + action_dim, curiosity_dim, hidden_dims, activation).to(self.device)
            self.parameters_module = self.model
        elif self.model_type == "rnd":
            self.model = make_mlp(curiosity_dim, emb_dim, hidden_dims, activation).to(self.device)
            self.target_model = make_mlp(curiosity_dim, emb_dim, hidden_dims, activation).to(self.device)
            for param in self.target_model.parameters():
                param.requires_grad = False
            self.parameters_module = self.model
        elif self.model_type == "disagreement":
            self.model = nn.ModuleList(
                [make_mlp(obs_dim + action_dim, curiosity_dim, hidden_dims, activation) for _ in range(ensemble_size)]
            ).to(self.device)
            self.parameters_module = self.model
        elif self.model_type == "neural_hash":
            self.model = HashAutoEncoder(curiosity_dim, hash_hidden_dim, code_dim).to(self.device)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(42)
            self.A = torch.randn((simhash_dim, code_dim), generator=generator, dtype=torch.float32, device=self.device)
            self.bin_count = torch.zeros(2**simhash_dim, dtype=torch.float32, device=self.device)
            self.parameters_module = self.model
        else:
            raise ValueError(f"Unsupported curiosity model_type: {self.model_type}")

        self.obs_act_normalization = bool(obs_act_normalization)
        self.curiosity_normalization = bool(curiosity_normalization)
        self.obs_act_normalizer = RunningNormalizer(obs_dim + action_dim).to(self.device)
        self.curiosity_normalizer = RunningNormalizer(curiosity_dim).to(self.device)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def state_dict(self) -> dict:
        state = {
            "model": self.model.state_dict(),
            "obs_act_normalizer": self.obs_act_normalizer.state_dict(),
            "curiosity_normalizer": self.curiosity_normalizer.state_dict(),
        }
        if self.model_type == "rnd":
            state["target_model"] = self.target_model.state_dict()
        if self.model_type == "neural_hash":
            state["A"] = self.A
            state["bin_count"] = self.bin_count
        return state

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.model.load_state_dict(state["model"], strict=strict)
        self.obs_act_normalizer.load_state_dict(state["obs_act_normalizer"], strict=strict)
        self.curiosity_normalizer.load_state_dict(state["curiosity_normalizer"], strict=strict)
        if self.model_type == "rnd" and "target_model" in state:
            self.target_model.load_state_dict(state["target_model"], strict=strict)
        if self.model_type == "neural_hash":
            self.A.copy_(state["A"].to(self.device))
            self.bin_count.copy_(state["bin_count"].to(self.device))

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor, action: torch.Tensor, curiosity: torch.Tensor) -> None:
        obs = obs.to(self.device, dtype=torch.float32)
        action = action.to(self.device, dtype=torch.float32)
        curiosity = curiosity.to(self.device, dtype=torch.float32)
        if self.obs_act_normalization:
            self.obs_act_normalizer.update(torch.cat([obs, action], dim=-1))
        if self.curiosity_normalization:
            self.curiosity_normalizer.update(curiosity)

    def _obs_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs.to(self.device, dtype=torch.float32), action.to(self.device, dtype=torch.float32)], dim=-1)
        return self.obs_act_normalizer(x) if self.obs_act_normalization else x

    def _curiosity(self, curiosity: torch.Tensor) -> torch.Tensor:
        x = curiosity.to(self.device, dtype=torch.float32)
        return self.curiosity_normalizer(x) if self.curiosity_normalization else x

    def compute_loss(self, obs: torch.Tensor, action: torch.Tensor, curiosity: torch.Tensor) -> torch.Tensor:
        if self.model_type == "prediction_error":
            pred = self.model(self._obs_action(obs, action))
            target = self._curiosity(curiosity)
            return (pred - target).square().mean()
        if self.model_type == "rnd":
            x = self._curiosity(curiosity)
            pred = self.model(x)
            with torch.no_grad():
                target = self.target_model(x)
            return (pred - target).square().mean()
        if self.model_type == "disagreement":
            x = self._obs_action(obs, action)
            target = self._curiosity(curiosity)
            preds = torch.stack([model(x) for model in self.model], dim=0)
            return (preds - target.unsqueeze(0)).square().mean()
        if self.model_type == "neural_hash":
            x = self._curiosity(curiosity)
            pred, code = self.model(x, noise_scale=self.hash_noise_scale)
            recon = (pred - x).square().mean()
            binary_reg = torch.minimum((1.0 - code).square(), code.square()).mean()
            return recon + self.hash_lambda_binary * binary_reg
        raise RuntimeError(f"Unsupported curiosity model_type: {self.model_type}")

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs: torch.Tensor, action: torch.Tensor, curiosity: torch.Tensor) -> torch.Tensor:
        if self.model_type == "prediction_error":
            pred = self.model(self._obs_action(obs, action))
            target = self._curiosity(curiosity)
            return (pred - target).square().mean(dim=-1)
        if self.model_type == "rnd":
            x = self._curiosity(curiosity)
            pred = self.model(x)
            target = self.target_model(x)
            return (pred - target).square().mean(dim=-1)
        if self.model_type == "disagreement":
            preds = torch.stack([model(self._obs_action(obs, action)) for model in self.model], dim=0)
            return preds.var(dim=0).mean(dim=-1)
        if self.model_type == "neural_hash":
            state_ids = self._hash_state_ids(self._curiosity(curiosity))
            counts = torch.bincount(state_ids, minlength=self.bin_count.numel()).to(self.bin_count.dtype)
            self.bin_count.add_(counts)
            return 1.0 / torch.sqrt(1.0 + self.bin_count[state_ids])
        raise RuntimeError(f"Unsupported curiosity model_type: {self.model_type}")

    @torch.no_grad()
    def _hash_state_ids(self, curiosity: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        _, code = self.model(curiosity, noise_scale=0.0)
        bits = (code >= 0.5).to(torch.long)
        signed = (bits * 2 - 1).to(torch.float32)
        sim_bits = (signed @ self.A.t() >= 0.0).to(torch.long)
        return bits_to_int(sim_bits).to(torch.long)
