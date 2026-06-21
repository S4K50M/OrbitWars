"""PPO policy and trainer for OrbitWars.

Architecture
------------
- Input:  planet_features [B, P, feat_dim],  alive_mask [B, P]
- 2-layer planet Transformer (pre-norm, batch_first MHA)
- Action head  : Linear(d_model → 1) per planet → masked softmax  →  target planet index
- Value head   : mean-pool alive planets → Linear → scalar
- PPO-clip update with GAE, gradient clipping, entropy bonus
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from interfaces import Policy, DecisionBatch

if TYPE_CHECKING:
    from rl_bridge import RLPolicyInput


# ── Planet Transformer ──────────────────────────────────────────────────────

class _PlanetLayer(nn.Module):
    """Pre-norm Transformer encoder layer (batch_first)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


@dataclass
class PolicyOutput:
    flow_matrix: Tensor  # [B, P, P] continuous values in [0, 1]
    value:       Tensor  # [B]
    flow_logstd: Tensor  # [1]

class OrbitWarsPolicy(nn.Module):
    def __init__(
        self,
        feat_dim: int = 16,
        d_model:  int = 128,
        n_heads:  int = 4,
        n_layers: int = 4,
        d_ff:     int = 512,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), 
            nn.LayerNorm(d_model), 
            nn.GELU(),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Unified Flow Head: Bilinear projection mapping source to target
        d_pair = d_model // 2
        self.src_proj = nn.Linear(d_model, d_pair)
        self.tgt_proj = nn.Linear(d_model, d_pair)
        self._scale = math.sqrt(d_pair)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        
        # Learnable standard deviation for continuous exploration (PPO)
        self.flow_logstd = nn.Parameter(torch.zeros(1))

    def forward(self, batch: DecisionBatch) -> PolicyOutput:
        inp = batch.policy_input
        x, alive, is_mine = inp.features, inp.alive_mask, inp.is_mine_mask
        B, P, _ = x.shape

        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=~alive)

        # Compute continuous pairwise flows
        q = self.src_proj(x)  # [B, P, d_pair]
        k = self.tgt_proj(x)  # [B, P, d_pair]
        flow_logits = torch.bmm(q, k.transpose(1, 2)) / self._scale  # [B, P, P]

        # Strategic Masking: Source must be mine, target must be alive and NOT mine
        src_ok = is_mine.unsqueeze(2).expand_as(flow_logits)
        tgt_ok = (alive & ~is_mine).unsqueeze(1).expand_as(flow_logits)
        eye = torch.eye(P, device=x.device, dtype=torch.bool).unsqueeze(0)
        
        # Apply mask: illegal moves are driven to -inf so sigmoid clears them to 0.0
        flow_logits = flow_logits.masked_fill(~src_ok | ~tgt_ok | eye, float("-inf"))
        flow_matrix = torch.sigmoid(flow_logits)

        # Value Prediction
        alive_f = alive.float().unsqueeze(-1)
        pooled = (x * alive_f).sum(1) / alive_f.sum(1).clamp(min=1.0)
        value = self.value_head(pooled).squeeze(-1)

        return PolicyOutput(flow_matrix=flow_matrix, value=value, flow_logstd=self.flow_logstd)

    def bc_forward(self, batch: DecisionBatch) -> Tensor:
        """Pass-through for BC that preserves grads across non-learner slots."""
        out = self.forward(batch)
        return out.flow_matrix

# ── Rollout buffer ──────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    obs_features: Tensor   # [T, B, P, feat_dim]
    obs_alive: Tensor      # [T, B, P]
    obs_is_mine: Tensor    # [T, B, P]
    actions: Tensor        # [T, B, P, P] (sampled matrix)
    log_probs: Tensor      # [T, B]
    values: Tensor         # [T, B]
    rewards: Tensor        # [T, B]
    dones: Tensor          # [T, B]

    @property
    def T(self) -> int:
        return self.actions.shape[0]

    @property
    def B(self) -> int:
        return self.actions.shape[1]

def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nd = (~dones[t]).float()
        delta = rewards[t] + gamma * nv * nd - values[t]
        gae = delta + gamma * lam * nd * gae
        advantages[t] = gae
    return advantages, advantages + values

# ── PPO Trainer ─────────────────────────────────────────────────────────────

class PPOTrainer:
    """Standard PPO-clip trainer for continuous matrix actions."""

    def __init__(
        self,
        policy: OrbitWarsPolicy,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        minibatch_envs: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.minibatch_envs = minibatch_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.total_env_steps = 0

    def update(self, buf: RolloutBuffer, last_value: Tensor, warmup: bool = False) -> dict:
        T, B = buf.T, buf.B
        adv, ret = compute_gae(
            buf.rewards, buf.values, buf.dones, last_value, self.gamma, self.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── ADD EXPLAINED VARIANCE TRACKING ──
        with torch.no_grad():
            ret_flat = ret.view(-1)
            val_flat = buf.values.view(-1)
            var_ret = ret_flat.var().item()
            var_res = (ret_flat - val_flat).var().item()
            ev = max(0.0, 1.0 - var_res / (var_ret + 1e-8))

        # Flatten T×B → N
        P = buf.obs_features.shape[2]
        feat_dim = buf.obs_features.shape[3]
        N = T * B
        
        feat       = buf.obs_features.view(N, P, feat_dim)
        alive      = buf.obs_alive.view(N, P)
        is_mine    = buf.obs_is_mine.view(N, P)
        actions    = buf.actions.view(N, P, P)
        olp        = buf.log_probs.view(N)
        ret_f      = ret.view(N)
        adv_f      = adv.view(N)

        metrics = dict(pg=0.0, vf=0.0, ent=0.0, clip_frac=0.0, explained_var=ev)
        mb = self.minibatch_envs  
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                idx = perm[start : start + mb]
                
                mb_feat = feat[idx]
                mb_alive = alive[idx]
                mb_is_mine = is_mine[idx]
                mb_actions = actions[idx]
                mb_olp = olp[idx]
                mb_ret = ret_f[idx]
                mb_adv = adv_f[idx]
                mb_B = len(idx)

                from rl_bridge import RLPolicyInput

                inp = RLPolicyInput(
                    features=mb_feat, 
                    alive_mask=mb_alive, 
                    is_mine_mask=mb_is_mine
                )
                
                mb_batch = DecisionBatch(
                    policy_input=inp,
                    env_index=torch.arange(mb_B, device=self.device),
                    rows_per_env=torch.ones(mb_B, dtype=torch.long, device=self.device),
                    metadata={},
                )

                out = self.policy(mb_batch)

                # ── Recompute Continuous Log Probs ──
                dist = torch.distributions.Normal(out.flow_matrix, out.flow_logstd.exp())
                
                # Active mask logic
                src_ok = mb_is_mine.unsqueeze(2).expand(mb_B, P, P)
                tgt_ok = (mb_alive & ~mb_is_mine).unsqueeze(1).expand(mb_B, P, P)
                eye = torch.eye(P, device=self.device, dtype=torch.bool).unsqueeze(0)
                mask = src_ok & tgt_ok & ~eye

                # Compute log prob and sum only over valid slots
                nlp = (dist.log_prob(mb_actions) * mask).sum(dim=(1, 2))
                ent = (dist.entropy() * mask).sum(dim=(1, 2)).mean()

                ratio = (nlp - mb_olp).exp()
                pg = -torch.min(
                    ratio * mb_adv,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_adv,
                ).mean()
                vf = F.mse_loss(out.value, mb_ret)
                
                # --- WARMUP SWITCH ---
                if warmup:
                    loss = self.vf_coef * vf
                else:
                    loss = pg + self.vf_coef * vf - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

                cf = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                metrics["pg"] += pg.item() if not warmup else 0.0
                metrics["vf"] += vf.item()
                metrics["ent"] += ent.item()
                metrics["clip_frac"] += cf
                n_updates += 1

        for k in metrics:
            metrics[k] /= max(n_updates, 1)
        self.total_env_steps += N
        return metrics

__all__ = ["OrbitWarsPolicy", "PolicyOutput", "RolloutBuffer", "PPOTrainer"]