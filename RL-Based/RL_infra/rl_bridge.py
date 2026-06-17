"""RL bridge: encoder, decoder, and optimised opponent for OrbitWars.

Key upgrades over v1
--------------------
- 12-feature planet encoder (vs 4) with orbital geometry
- ThreadPoolExecutor opponent — parallel rule-agent calls over all B envs
- Fixed decode_actions: produces real Move lists from sampled target planet
- `act_seats` pre-copies all obs to CPU once, then fans out via threads
"""
from __future__ import annotations

import math
import random
import dataclasses
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch
from torch import Tensor

from interfaces import ObservationEncoder, Opponent, DecisionBatch, Move
from engine import TorchOrbitWarsEnv
import RuleAgent as default_agent


# ── Constants (mirrored from engine) ────────────────────────────────────────
CENTER = 50.0
ROT_RADIUS_LIMIT = 50.0


# ── Feature dataclass ────────────────────────────────────────────────────────

# ── Feature dataclass ────────────────────────────────────────────────────────

@dataclass
class RLPolicyInput:
    features: Tensor      # [B, P, 14]
    alive_mask: Tensor    # [B, P] bool
    is_mine_mask: Tensor  # [B, P] bool

# ── Observation encoder ──────────────────────────────────────────────────────

def _encode_features(
    engine: TorchOrbitWarsEnv,
    player_per_env: Tensor,
) -> RLPolicyInput:
    """Fully batched 14-feature planet encoder."""
    B, P = engine.B, engine.P
    device = engine.device
    ft = engine.dtype

    pid = engine.planet_id.to(ft)          # [B, P]
    owner = engine.planet_owner.to(ft)
    px = engine.planet_x
    py = engine.planet_y
    ships = engine.planet_ships.to(ft)
    prod = engine.planet_production.to(ft)
    ix = engine.planet_initial_x
    iy = engine.planet_initial_y
    radius = engine.planet_radius

    alive = pid >= 0.0                     # [B, P]
    my_seat = player_per_env.view(B, 1).to(ft)

    is_mine    = (alive & (owner == my_seat))
    is_enemy   = (alive & (owner >= 0) & (owner != my_seat)).to(ft)
    is_neutral = (alive & (owner < 0)).to(ft)
    ships_norm = (ships / 1000.0).clamp(max=1.0)
    prod_norm  = (prod / 5.0).clamp(max=1.0)

    # Orbital geometry
    dx0 = ix - CENTER
    dy0 = iy - CENTER
    orb_r = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    orb_a0 = torch.atan2(dy0, dx0)
    is_orb = (alive & ((orb_r + radius) < ROT_RADIUS_LIMIT) & (orb_r > 0.5)).to(ft)
    dist_sun_norm = (orb_r / ROT_RADIUS_LIMIT).clamp(max=1.0)

    # Current angle
    dx = px - CENTER
    dy = py - CENTER
    cur_angle = torch.atan2(dy, dx)
    
    # ── NEW FEATURES (13 & 14): Reachability & Proximity ──
    dx_all = px.unsqueeze(2) - px.unsqueeze(1)   # [B, P, P]
    dy_all = py.unsqueeze(2) - py.unsqueeze(1)
    dist_all = torch.sqrt((dx_all**2 + dy_all**2).clamp(min=0))  # [B, P, P]
    INF = 999.0
    
    # Distance to nearest owned planet
    owned_dist = dist_all.masked_fill(~is_mine.unsqueeze(2).expand_as(dist_all), INF)
    dist_nearest_owned = owned_dist.amin(dim=1)  # [B, P]
    
    reachable = (dist_nearest_owned < 35.0).to(ft)  # ~15 turns at speed 2.3
    dist_nearest_norm = (dist_nearest_owned / 70.0).clamp(max=1.0)

    features = torch.stack([
        is_mine.to(ft), is_enemy, is_neutral,
        ships_norm, prod_norm, dist_sun_norm,
        torch.sin(cur_angle), torch.cos(cur_angle),
        torch.sin(orb_a0), torch.cos(orb_a0),
        is_orb, alive.to(ft),
        reachable, dist_nearest_norm
    ], dim=-1)   # [B, P, 14]

    return RLPolicyInput(features=features, alive_mask=alive, is_mine_mask=is_mine)


class OrbitWarsEncoder(ObservationEncoder):
    """12-feature planet encoder + planet-to-planet action decoder."""

    def __init__(self, movement_horizon: int = 15):
        self.horizon = movement_horizon

    def encode(
        self, engine: TorchOrbitWarsEnv, player_per_env: Tensor
    ) -> DecisionBatch:
        B = engine.B
        device = engine.device

        inp = _encode_features(engine, player_per_env)

        # Snapshot coords for decode (cheap — stays on GPU)
        meta = {
            "planet_x":     engine.planet_x.clone(),
            "planet_y":     engine.planet_y.clone(),
            "planet_owner": engine.planet_owner.clone(),
            "planet_ships": engine.planet_ships.clone(),
            "planet_id":    engine.planet_id.clone(),    # needed for move encoding
            "planet_alive": (engine.planet_id >= 0),
            "player_per_env": player_per_env.clone(),
        }

        return DecisionBatch(
            policy_input=inp,
            env_index=torch.arange(B, device=device),
            rows_per_env=torch.ones(B, dtype=torch.long, device=device),
            metadata=meta,
        )

    def decode_actions(
        self,
        batch: DecisionBatch,
        target_index: Tensor,
        num_envs: int,
    ) -> list[list[Move]]:
        """
        Unpack flat (src_idx * P + tgt_idx) action → Move list.
        Sends 75% of source ships toward the target (min 5 ships).
        """
        meta   = batch.metadata
        px     = meta["planet_x"]        # [B, P]
        py     = meta["planet_y"]
        owner  = meta["planet_owner"]    # [B, P] int64
        ships  = meta["planet_ships"]    # [B, P] int64
        pids   = meta["planet_id"]       # [B, P] int64
        alive  = meta["planet_alive"]    # [B, P] bool
        seats  = meta["player_per_env"]  # [B]    int64
        B, P   = px.shape

        # ── Unpack flat action ────────────────────────────────────────────
        flat = target_index.clamp(0, P * P - 1)
        src_idx = flat // P    # [B]
        tgt_idx = flat % P     # [B]

        src_idx = src_idx.clamp(0, P - 1)
        tgt_idx = tgt_idx.clamp(0, P - 1)

        # ── Source properties ─────────────────────────────────────────────
        src_pid  = pids.gather(1, src_idx.unsqueeze(1)).squeeze(1)   # [B]
        src_ships = ships.gather(1, src_idx.unsqueeze(1)).squeeze(1) # [B]
        src_x    = px.gather(1, src_idx.unsqueeze(1)).squeeze(1)
        src_y    = py.gather(1, src_idx.unsqueeze(1)).squeeze(1)
        src_own  = owner.gather(1, src_idx.unsqueeze(1)).squeeze(1)  # [B]

        # ── Target properties ─────────────────────────────────────────────
        tgt_x = px.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)
        tgt_y = py.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)
        tgt_own = owner.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)   # [B]

        # ── Angle: direct aim (engine handles orbital lead for us) ────────
        angle = torch.atan2(tgt_y - src_y, tgt_x - src_x)

        # ── Ship count: 75% of source ships, minimum 5 ───────────────────
        n_send = (src_ships.float() * 0.75).floor().long()

        # ── Validity ──────────────────────────────────────────────────────
        src_is_mine = (src_own == seats)               # source is owned by learner
        tgt_is_enemy = (tgt_own != seats)              # target is NOT ours (Neutral or Enemy)
        tgt_is_alive = alive.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)
        has_ships = (n_send >= 5)
        not_same = (src_idx != tgt_idx)

        valid = src_is_mine & tgt_is_enemy & tgt_is_alive & has_ships & not_same

        # ── CPU transfer (single call) ────────────────────────────────────
        valid_cpu   = valid.cpu().tolist()
        src_pid_cpu = src_pid.cpu().tolist()
        angle_cpu   = angle.cpu().tolist()
        n_send_cpu  = n_send.cpu().tolist()

        moves: list[list[Move]] = []
        for i in range(B):
            if valid_cpu[i]:
                moves.append([[int(src_pid_cpu[i]), float(angle_cpu[i]), int(n_send_cpu[i])]])
            else:
                moves.append([])
        return moves

# ── Threaded multi-agent opponent ────────────────────────────────────────────

class MultiAgentOpponent(Opponent):
    """
    Pool of rule-based agents with ThreadPoolExecutor for parallelism.

    The 1024-env serial Python loop is replaced with `max_workers` threads
    running agent calls concurrently.  Each env has its own Memory so there
    is no shared mutable state between threads.
    """

    def __init__(
        self,
        agent_modules: list[Any],
        max_workers: int = 16,
    ):
        self.agent_modules = agent_modules
        self.memories: dict[int, Any] = {}
        self.active_agents: dict[int, Any] = {}
        self.rng = random.Random()
        self.max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    # ── internal: process one env ──────────────────────────────────────────

    def _act_one(
        self,
        i: int,
        single_obs: dict,
        step_val: int,
    ) -> list[Move]:
        """Called in a thread. Returns move list for env i."""
        # Assign agent at episode start
        if i not in self.active_agents or step_val == 0:
            self.active_agents[i] = self.rng.choice(self.agent_modules)
            self.memories[i] = self.active_agents[i].ProducerLiteMemory()

        agent = self.active_agents[i]
        mem   = self.memories[i]

        if mem.cached_player_count is None:
            mem.cached_player_count = agent.largest_initial_player_count(single_obs)

        config = agent._config_for(mem.cached_player_count)
        sparse = agent.run_turn(
            single_obs,
            config=config,
            player_count=int(mem.cached_player_count),
            memory=mem,
        )

        counts = int(sparse["counts"].item())
        env_moves: list[Move] = []
        for c in range(counts):
            f_id = int(sparse["from_planet_id"][c].item())
            ang  = float(sparse["angle"][c].item())
            sh   = int(sparse["num_ships"][c].item())
            if sh > 0:
                env_moves.append([f_id, ang, sh])
        return env_moves

    # ── Opponent.act ──────────────────────────────────────────────────────

    def act(
        self, engine: TorchOrbitWarsEnv, seat_per_env: Tensor
    ) -> list[list[Move]]:
        B = engine.B

        # Pre-snapshot all batched obs to CPU in ONE transfer
        p_cpu = engine.planet_id.cpu()
        owner_cpu = engine.planet_owner.cpu()
        px_cpu = engine.planet_x.cpu()
        py_cpu = engine.planet_y.cpu()
        pr_cpu = engine.planet_radius.cpu()
        ps_cpu = engine.planet_ships.cpu()
        pp_cpu = engine.planet_production.cpu()
        pix_cpu = engine.planet_initial_x.cpu()
        piy_cpu = engine.planet_initial_y.cpu()

        f_id_cpu  = engine.fleet_id.cpu()
        f_own_cpu = engine.fleet_owner.cpu()
        f_x_cpu   = engine.fleet_x.cpu()
        f_y_cpu   = engine.fleet_y.cpu()
        f_ang_cpu = engine.fleet_angle.cpu()
        f_fp_cpu  = engine.fleet_from_pid.cpu()
        f_sh_cpu  = engine.fleet_ships.cpu()

        step_cpu  = engine.step_count.cpu()
        angv_cpu  = engine.angular_velocity.cpu()
        nfid_cpu  = engine.next_fleet_id.cpu()
        seat_cpu  = seat_per_env.cpu()

        dtype = engine.dtype

        def make_obs(i: int) -> dict:
            def _p(t): return t[i].to(dtype)
            pl = torch.stack([
                _p(p_cpu), _p(owner_cpu), _p(px_cpu), _p(py_cpu),
                _p(pr_cpu), _p(ps_cpu), _p(pp_cpu),
            ], dim=-1)
            ipl = pl.clone()
            ipl[:, 2] = pix_cpu[i].to(dtype)
            ipl[:, 3] = piy_cpu[i].to(dtype)
            fl = torch.stack([
                _p(f_id_cpu), _p(f_own_cpu), _p(f_x_cpu), _p(f_y_cpu),
                _p(f_ang_cpu), _p(f_fp_cpu), _p(f_sh_cpu),
            ], dim=-1)
            return {
                "planets":          pl,
                "fleets":           fl,
                "initial_planets":  ipl,
                "step":             step_cpu[i].to(dtype),
                "angular_velocity": angv_cpu[i].to(dtype),
                "next_fleet_id":    nfid_cpu[i].to(dtype),
                "player_count":     torch.tensor(engine.cfg.num_agents, dtype=dtype),
                "player":           seat_cpu[i].to(dtype),
            }

        results: list[list[Move]] = [[] for _ in range(B)]
        futures = {}
        for i in range(B):
            if engine._done_cpu[i]:
                continue
            obs_i   = make_obs(i)
            step_i  = int(step_cpu[i].item())
            fut = self._pool.submit(self._act_one, i, obs_i, step_i)
            futures[fut] = i

        for fut, i in futures.items():
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = []

        return results

    def close(self):
        self._pool.shutdown(wait=False)



class GreedyVectorOpponent(Opponent):
    """
    Fully vectorised greedy opponent — ZERO Python per-env loop.

    Strategy (all ops on [B, P] tensors on the engine device):
    1. Source : owned planet with the most ships.
    2. Target : alive non-owned planet nearest to source.
    3. Move   : send floor(src_ships * send_frac) ships at the computed angle.
    Falls back to an empty move list for envs where no valid move exists.
    """

    def __init__(self, send_frac: float = 0.55):
        self.send_frac = send_frac

    def act(
        self, engine: TorchOrbitWarsEnv, seat_per_env: Tensor
    ) -> list[list[Move]]:
        B, P = engine.B, engine.P
        device = engine.device

        alive  = engine.planet_id >= 0                     # [B, P]
        owner  = engine.planet_owner                       # [B, P] int64
        ships  = engine.planet_ships                       # [B, P] int64
        px     = engine.planet_x                           # [B, P] float
        py     = engine.planet_y
        pids   = engine.planet_id                          # [B, P] int64
        seat   = seat_per_env.unsqueeze(1)                 # [B, 1]

        # ── source: owned planet with most ships ──────────────────────────
        owned_mask = alive & (owner == seat)               # [B, P]
        src_ships  = ships.masked_fill(~owned_mask, -1)   # [B, P]
        best_src   = src_ships.argmax(dim=1)               # [B]
        has_src    = owned_mask.any(dim=1)                 # [B]

        src_x  = px.gather(1, best_src.unsqueeze(1)).squeeze(1)  # [B]
        src_y  = py.gather(1, best_src.unsqueeze(1)).squeeze(1)
        src_sh = ships.gather(1, best_src.unsqueeze(1)).squeeze(1)
        src_pid = pids.gather(1, best_src.unsqueeze(1)).squeeze(1)

        # ── target: closest alive non-owned planet ────────────────────────
        non_owned = alive & (owner != seat)                # [B, P]
        dx = px - src_x.unsqueeze(1)                       # [B, P]
        dy = py - src_y.unsqueeze(1)
        dist2 = dx * dx + dy * dy
        dist2_masked = dist2.masked_fill(~non_owned, float("inf"))
        best_tgt = dist2_masked.argmin(dim=1)              # [B]
        has_tgt  = non_owned.any(dim=1)                    # [B]

        tgt_x = px.gather(1, best_tgt.unsqueeze(1)).squeeze(1)
        tgt_y = py.gather(1, best_tgt.unsqueeze(1)).squeeze(1)

        # ── move parameters ───────────────────────────────────────────────
        angle  = torch.atan2(tgt_y - src_y, tgt_x - src_x)        # [B]
        n_send = (src_sh.float() * self.send_frac).floor().long()  # [B]

        # ── build move lists (single batch .cpu() call) ───────────────────
        valid = has_src & has_tgt & (n_send > 0) & ~engine.done   # [B]
        valid_cpu  = valid.cpu().tolist()
        src_pid_cpu = src_pid.cpu().tolist()
        angle_cpu   = angle.cpu().tolist()
        n_send_cpu  = n_send.cpu().tolist()

        moves: list[list[Move]] = []
        for i in range(B):
            if valid_cpu[i]:
                moves.append([[int(src_pid_cpu[i]), float(angle_cpu[i]), int(n_send_cpu[i])]])
            else:
                moves.append([])
        return moves


__all__ = [
    "OrbitWarsEncoder",
    "MultiAgentOpponent",
    "GreedyVectorOpponent",
    "RLPolicyInput",
]