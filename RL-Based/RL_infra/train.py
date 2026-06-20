"""OrbitWars PPO training script.
Hardware: targets cuda:4 (NVIDIA RTX A6000, ~48 GB free).

Usage
-----
    python3 RL-Based/RL_infra/train.py

Key design
----------
- 1024 parallel envs (batched GPU engine)
- Multi-rule-agent opponent pool via ThreadPoolExecutor (16 workers)
- PPO with 128-step rollouts, GAE(γ=0.99, λ=0.95), 4 epochs per update
- Checkpoint saved every 50 updates to checkpoints/policy_<step>.pt
- SPS printed every update; rewards logged when episodes complete
"""
from __future__ import annotations

import os
import sys
import time
import math
import json
import logging
from pathlib import Path

# Fragmentation fix for large batch attention
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn

# Make RL_infra importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from engine import TorchEngineConfig
from vector_env import GpuVectorEnv
from rl_bridge import OrbitWarsEncoder, MultiAgentOpponent, GreedyVectorOpponent, RLPolicyInput, SelfPlayOpponent
from ppo_policy import OrbitWarsPolicy, PPOTrainer, RolloutBuffer
from interfaces import DecisionBatch

import Experiments.triMultiSearch as agent2
import Experiments.MultiAgent as agent3
import RuleAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ── Hyper-parameters ────────────────────────────────────────────────────────

NUM_ENVS          = 1024
NUM_AGENTS        = 2
ROLLOUT_STEPS     = 128       # steps per update (128 × 1024 = 131,072 env-steps)
TOTAL_UPDATES     = 10_000    # training budget
FEAT_DIM          = 16        # updated to match new encoder
D_MODEL           = 96        # bumped capacity
N_HEADS           = 4
N_LAYERS          = 3         # bumped capacity

LR                = 3e-5      # Lowered for stable transformer learning
CLIP_EPS          = 0.2
VF_COEF           = 0.5
ENT_COEF          = 0.005     # Decreased to prevent entropy trap with massive action space
MAX_GRAD_NORM     = 1.0       # Increased gradient allowance
N_EPOCHS          = 1         # Prevents severe overfitting on the same data
MINIBATCH_ENVS    = 2048      # Adapted for N_EPOCHS=1

GAMMA             = 0.99
GAE_LAMBDA        = 0.95

CHECKPOINT_DIR    = Path("checkpoints")
CHECKPOINT_EVERY  = 50        # updates
LOG_EVERY         = 5         # updates
SYNC_OPPONENT_EVERY = 50      # Updates before syncing the self-play opponent


def pick_device() -> torch.device:
    """Pick the GPU with most free memory (prefers cuda:4/5/7 which are idle)."""
    if not torch.cuda.is_available():
        log.warning("CUDA not available — running on CPU (will be slow).")
        return torch.device("cpu")
    # Prefer GPUs 4, 5, 7 (known-free from nvidia-smi)
    preferred = [4, 5, 7, 3, 1, 0]
    n = torch.cuda.device_count()
    for idx in preferred:
        if idx < n:
            free, total = torch.cuda.mem_get_info(idx)
            if free > 10 * 1024**3:  # at least 10 GB free
                log.info("Selected cuda:%d  (%.1f GB free)", idx, free / 1024**3)
                return torch.device(f"cuda:{idx}")
    return torch.device("cuda:0")


def make_env(device: torch.device, encoder: OrbitWarsEncoder) -> GpuVectorEnv:
    engine_cfg = TorchEngineConfig(
        num_envs=NUM_ENVS, num_agents=NUM_AGENTS, device=device,
    )
    opponent = SelfPlayOpponent(
        encoder=encoder,
        device=device,
        greedy_mix_prob=0.30,    # 30% greedy keeps training diverse
    )
    return GpuVectorEnv(
        engine_cfg=engine_cfg,
        encoder=encoder,
        opponent=opponent,
        learner_seat=0,
        capture_shaping_coef=0.02, # Shaping coefficient linked!
    )

def make_policy(device: torch.device) -> OrbitWarsPolicy:
    return OrbitWarsPolicy(
        feat_dim=FEAT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ).to(device)


@torch.no_grad()
def collect_rollout(
    env: GpuVectorEnv,
    policy: OrbitWarsPolicy,
    batch: DecisionBatch,
    rollout_steps: int,
    device: torch.device,
) -> tuple[RolloutBuffer, DecisionBatch, float]:
    """Collect rollout_steps of experience. Returns buffer, next_batch, mean_ep_reward."""
    B = env.num_envs
    P = batch.policy_input.features.shape[1]
    feat_dim = batch.policy_input.features.shape[2]

    obs_feat   = torch.zeros(rollout_steps, B, P, feat_dim, device=device)
    obs_alive  = torch.zeros(rollout_steps, B, P, dtype=torch.bool, device=device)
    actions    = torch.zeros(rollout_steps, B, dtype=torch.long, device=device)
    log_probs  = torch.zeros(rollout_steps, B, device=device)
    values     = torch.zeros(rollout_steps, B, device=device)
    rewards    = torch.zeros(rollout_steps, B, device=device)
    dones      = torch.zeros(rollout_steps, B, dtype=torch.bool, device=device)

    ep_rewards: list[float] = []

    for t in range(rollout_steps):
        inp: RLPolicyInput = batch.policy_input
        obs_feat[t]  = inp.features
        obs_alive[t] = inp.alive_mask
        
        # Notice we do NOT track obs_is_mine here to keep the RolloutBuffer clean.
        # We will dynamically recreate it in the PPOTrainer using feature [0].

        out = policy(batch)
        dist = torch.distributions.Categorical(logits=out.target_logits)
        act  = dist.sample()
        lp   = dist.log_prob(act)
        val  = out.value

        result = env.step(batch, act)

        actions[t]   = act
        log_probs[t] = lp
        values[t]    = val
        rewards[t]   = result.reward
        dones[t]     = result.done

        # Track episode rewards
        done_idx = result.done.nonzero(as_tuple=True)[0]
        for i in done_idx.tolist():
            ep_rewards.append(result.reward[i].item())

        batch = result.next_batch

    mean_ep_rew = float(sum(ep_rewards) / len(ep_rewards)) if ep_rewards else 0.0
    buf = RolloutBuffer(
        obs_features=obs_feat,
        obs_alive=obs_alive,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
    )
    return buf, batch, mean_ep_rew


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    device  = pick_device()
    
    # Encoder initialized FIRST so the env can inject it into the Self-Play opponent
    encoder = OrbitWarsEncoder(movement_horizon=15)
    env     = make_env(device, encoder)
    policy  = make_policy(device)

    policy = torch.compile(policy, mode="reduce-overhead")

    trainer = PPOTrainer(
        policy=policy,
        lr=LR,
        clip_eps=CLIP_EPS,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        n_epochs=N_EPOCHS,
        minibatch_envs=MINIBATCH_ENVS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        device=device,
    )

    # Disable param printing for compiled models to avoid torch.compile warnings
    log.info("Collecting rollouts: T=%d × B=%d = %d env-steps per update",
             ROLLOUT_STEPS, NUM_ENVS, ROLLOUT_STEPS * NUM_ENVS)

    log.info("Resetting environments …")
    batch = env.reset()
    log.info("Feature shape: %s", tuple(batch.policy_input.features.shape))

    # ── Warmup: discard first 3 steps to let CUDA JIT settle ──────────────
    for _ in range(3):
        with torch.no_grad():
            out = policy(batch)
            dist = torch.distributions.Categorical(logits=out.target_logits)
            act  = dist.sample()
        result = env.step(batch, act)
        batch  = result.next_batch

    total_env_steps = 0
    t_start = time.perf_counter()
    sps_window: list[float] = []

    for update in range(1, TOTAL_UPDATES + 1):
        t0 = time.perf_counter()

        buf, batch, ep_rew = collect_rollout(
            env, policy, batch, ROLLOUT_STEPS, device
        )

        # Bootstrap value for last state
        with torch.no_grad():
            last_val = policy(batch).value   # [B]

        metrics = trainer.update(buf, last_val)
        total_env_steps += ROLLOUT_STEPS * NUM_ENVS

        elapsed  = time.perf_counter() - t0
        sps      = (ROLLOUT_STEPS * NUM_ENVS) / elapsed
        sps_window.append(sps)
        if len(sps_window) > 20:
            sps_window.pop(0)
        smoothed_sps = sum(sps_window) / len(sps_window)

        # ── NEW: Sync self-play opponent ──
        if update % SYNC_OPPONENT_EVERY == 0:
            env.update_opponent_from(policy)
            log.info("Synced self-play opponent snapshot at update %d", update)

        if update % 100 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if update % LOG_EVERY == 0:
            wall = time.perf_counter() - t_start
            log.info(
                "Update %4d/%d | env-steps %s | SPS %.0f (avg %.0f) | "
                "pg %.4f  vf %.4f  ent %.4f  clip %.2f  ev %.3f | ep_rew %.3f | wall %s",
                update, TOTAL_UPDATES,
                f"{total_env_steps:,}",
                sps, smoothed_sps,
                metrics["pg"], metrics["vf"], metrics["ent"], metrics["clip_frac"],
                metrics.get("explained_var", 0.0), ep_rew,
                f"{wall:.0f}s",
            )

        if update % CHECKPOINT_EVERY == 0:
            path = CHECKPOINT_DIR / f"policy_{total_env_steps}.pt"
            torch.save({
                "update": update,
                "total_env_steps": total_env_steps,
                "policy_state": policy.state_dict(),
                "optimizer_state": trainer.opt.state_dict(),
                "metrics": metrics,
            }, path)
            log.info("Saved checkpoint → %s", path)

    log.info("Training complete. Total env-steps: %s", f"{total_env_steps:,}")
    env.close()

if __name__ == "__main__":
    main()