from __future__ import annotations

import math

import torch
from torch import nn


class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach()
        batch_count = torch.tensor(float(x.shape[0]), dtype=self.count.dtype, device=self.count.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if float(self.count.item()) == 0.0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var.clamp_min(self.eps))
            self.count.copy_(batch_count)
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total.clamp_min(1.0)
        self.mean.add_(delta * batch_count / total.clamp_min(1.0))
        self.var.copy_((m_2 / total.clamp_min(1.0)).clamp_min(self.eps))
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class HashAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, code_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.code = nn.Linear(hidden_dim, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor, noise_scale: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        b = torch.sigmoid(self.code(h))
        if noise_scale > 0.0:
            noise = (torch.rand_like(b) * 2.0 - 1.0) * float(noise_scale)
            b_dec = (b + noise).clamp(0.0, 1.0)
        else:
            b_dec = b
        return self.dec(b_dec), b


@torch.no_grad()
def bits_to_int(bits01: torch.Tensor) -> torch.Tensor:
    bits = bits01.to(torch.long)
    shifts = torch.arange(bits.shape[1], dtype=torch.long, device=bits.device)
    return (bits << shifts).sum(dim=1)


@torch.no_grad()
def state_id_entropy(state_ids: torch.Tensor, num_states: int) -> torch.Tensor:
    counts = torch.bincount(state_ids.to(torch.long), minlength=int(num_states)).float()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return entropy / math.log(float(num_states))


def infer_simhash_dim(num_key_states: int) -> int:
    if num_key_states < 2 or num_key_states & (num_key_states - 1):
        raise ValueError(f"num_key_states must be a power of two for hash state banks, got {num_key_states}")
    return num_key_states.bit_length() - 1


class LearnedHashStateBank:
    def __init__(
        self,
        *,
        num_key_states: int,
        feature_dim: int,
        buffer_size: int,
        num_hand_keypoints: int,
        num_object_bins: int,
        device: torch.device,
        code_dim: int = 256,
        simhash_dim: int | None = None,
        hidden_dim: int = 512,
        noise_scale: float = 0.3,
        lambda_binary: float = 10.0,
        ae_lr: float = 3e-4,
        ae_update_steps: int = 5,
        ae_update_freq: int = 16,
        ae_num_minibatches: int = 8,
        seed: int = 0,
    ):
        self.S = int(num_key_states)
        self.F = int(feature_dim)
        self.B = int(buffer_size)
        self.L = int(num_hand_keypoints)
        self.K = int(num_object_bins)
        self.device = device
        self.code_dim = int(code_dim)
        self.simhash_dim = infer_simhash_dim(self.S) if simhash_dim is None else int(simhash_dim)
        self.noise_scale = float(noise_scale)
        self.lambda_binary = float(lambda_binary)
        self.ae_update_steps = int(ae_update_steps)
        self.ae_update_freq = max(1, int(ae_update_freq))
        self.ae_num_minibatches = max(1, int(ae_num_minibatches))

        self.counts = torch.zeros((self.S, self.L, self.K), dtype=torch.float32, device=self.device)
        self.ae = HashAutoEncoder(self.F, int(hidden_dim), self.code_dim).to(self.device)
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=float(ae_lr))
        self.normalizer = RunningNormalizer(self.F).to(self.device)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        self.A = torch.randn(
            (self.simhash_dim, self.code_dim), generator=generator, dtype=torch.float32, device=self.device
        )

        self.buffer = torch.zeros((self.B, self.F), dtype=torch.float32, device=self.device)
        self.buf_n = 0
        self.buf_pos = 0
        self.step_count = 0
        self.update_enabled = True
        self.last_recon = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.last_reg = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.last_entropy = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def reset(self, *, reset_counters: bool = False) -> None:
        self.buf_n = 0
        self.buf_pos = 0
        if reset_counters:
            self.counts.zero_()

    @torch.no_grad()
    def push(self, feats: torch.Tensor) -> None:
        if not self.update_enabled:
            return
        feats = feats.to(self.device, dtype=torch.float32)
        self.normalizer.update(feats)
        n = int(feats.shape[0])
        if n >= self.B:
            self.buffer.copy_(feats[-self.B :])
            self.buf_n = self.B
            self.buf_pos = 0
        else:
            end = self.buf_pos + n
            if end <= self.B:
                self.buffer[self.buf_pos : end] = feats
            else:
                first = self.B - self.buf_pos
                self.buffer[self.buf_pos :] = feats[:first]
                self.buffer[: end - self.B] = feats[first:]
            self.buf_pos = end % self.B
            self.buf_n = min(self.B, self.buf_n + n)

        self.step_count += 1
        if self.buf_n >= max(self.ae_num_minibatches, 2) and self.step_count % self.ae_update_freq == 0:
            with torch.inference_mode(False), torch.enable_grad(), torch.set_grad_enabled(True):
                self._update_autoencoder()

    @torch.no_grad()
    def assign(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.to(self.device, dtype=torch.float32)
        x = self.normalizer(feats)
        self.ae.eval()
        _, code = self.ae(x, noise_scale=0.0)
        bits = (code >= 0.5).to(torch.long)
        signed = (bits * 2 - 1).to(torch.float32)
        sim_bits = (signed @ self.A.t() >= 0.0).to(torch.long)
        state_ids = (bits_to_int(sim_bits) % self.S).to(torch.long)
        self.last_entropy = state_id_entropy(state_ids, self.S)
        return state_ids

    def _update_autoencoder(self) -> None:
        n = int(self.buf_n)
        if n < 2:
            return
        x_all = self.normalizer(self.buffer[:n])
        self.ae.train()
        recon = self.last_recon
        reg = self.last_reg
        for _ in range(max(1, self.ae_update_steps)):
            perm = torch.randperm(n, device=self.device)
            for idx in torch.chunk(perm, self.ae_num_minibatches):
                if idx.numel() == 0:
                    continue
                x = x_all.index_select(0, idx)
                x_hat, code = self.ae(x, noise_scale=self.noise_scale)
                recon = (x_hat - x).square().mean()
                reg = torch.minimum((1.0 - code).square(), code.square()).mean()
                loss = recon + self.lambda_binary * reg
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
        self.last_recon = recon.detach()
        self.last_reg = reg.detach()
        self.ae.eval()

    @torch.no_grad()
    def add_contacts(
        self,
        *,
        state_ids: torch.Tensor,
        contact_mask: torch.Tensor,
        contact_bins: torch.Tensor,
    ) -> None:
        state_ids = state_ids.to(self.device, dtype=torch.long).clamp(0, self.S - 1)
        contact_mask = contact_mask.to(self.device, dtype=torch.bool)
        contact_bins = contact_bins.to(self.device, dtype=torch.long).clamp(0, self.K - 1)
        if not contact_mask.any():
            return
        env_ids, kp_ids = torch.nonzero(contact_mask, as_tuple=True)
        cluster_ids = contact_bins[env_ids, kp_ids]
        linear = ((state_ids[env_ids] * self.L + kp_ids) * self.K + cluster_ids).to(torch.long)
        counts = torch.bincount(linear, minlength=self.S * self.L * self.K).view(self.S, self.L, self.K)
        self.counts.add_(counts.to(self.counts.dtype))

    @torch.no_grad()
    def get_metrics(self) -> dict[str, torch.Tensor]:
        return {
            "hash_recon_loss": self.last_recon.detach(),
            "hash_binary_reg": self.last_reg.detach(),
            "stateid_entropy": self.last_entropy.detach(),
        }
