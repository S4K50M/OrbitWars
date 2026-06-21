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

import copy
from torch.distributions import Categorical

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
    """Fully batched 16-feature planet encoder."""
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
    
    # Reachability & Proximity
    dx_all = px.unsqueeze(2) - px.unsqueeze(1)
    dy_all = py.unsqueeze(2) - py.unsqueeze(1)
    dist_all = torch.sqrt((dx_all**2 + dy_all**2).clamp(min=0))
    INF = 999.0
    
    owned_dist = dist_all.masked_fill(~is_mine.unsqueeze(2).expand_as(dist_all), INF)
    dist_nearest_owned = owned_dist.amin(dim=1)
    
    reachable = (dist_nearest_owned < 35.0).to(ft)
    dist_nearest_norm = (dist_nearest_owned / 70.0).clamp(max=1.0)

    # ── NEW FEATURES (15 & 16): Global Game Phase & Economy ──
    step_norm = (engine.step_count.to(ft) / 500.0).view(B, 1).expand(B, P)
    
    total_prod = prod.sum(dim=1, keepdim=True).clamp(min=1.0)
    my_prod    = (prod * is_mine.to(ft)).sum(dim=1, keepdim=True)
    prod_ratio = (my_prod / total_prod).expand(B, P)

    features = torch.stack([
        is_mine.to(ft), is_enemy, is_neutral,
        ships_norm, prod_norm, dist_sun_norm,
        torch.sin(cur_angle), torch.cos(cur_angle),
        torch.sin(orb_a0), torch.cos(orb_a0),
        is_orb, alive.to(ft),
        reachable, dist_nearest_norm,
        step_norm, prod_ratio # <--- Added here
    ], dim=-1)   # [B, P, 16]

    return RLPolicyInput(features=features, alive_mask=alive, is_mine_mask=is_mine)

_FLEET_SPEED = 2.3   # approximate; close enough for decoder aiming

def _intercept_angle(
    src_x:   Tensor,   # [B]
    src_y:   Tensor,   # [B]
    tgt_x:   Tensor,   # [B] current position
    tgt_y:   Tensor,   # [B]
    meta:    dict,
    tgt_idx: Tensor,   # [B] planet index
) -> Tensor:           # [B] angle in radians
    """
    For orbiting targets: lead the planet to where it will be when
    the fleet arrives. One iteration of linear approximation.
    For static targets: direct angle.
    """
    orb_r     = meta["planet_orb_r"]       # [B, P]
    cur_angle = meta["planet_cur_angle"]   # [B, P]
    ang_vel   = meta["angular_velocity"]   # [B]

    tgt_orb_r = orb_r.gather(1,     tgt_idx.unsqueeze(1)).squeeze(1)   # [B]
    tgt_cur_a = cur_angle.gather(1,  tgt_idx.unsqueeze(1)).squeeze(1)  # [B]

    dx   = tgt_x - src_x
    dy   = tgt_y - src_y
    dist = (dx * dx + dy * dy).sqrt().clamp(min=1e-3)
    t_est = dist / _FLEET_SPEED

    future_a = tgt_cur_a + ang_vel * t_est
    future_x = CENTER + tgt_orb_r * torch.cos(future_a)
    future_y = CENTER + tgt_orb_r * torch.sin(future_a)

    intercept = torch.atan2(future_y - src_y, future_x - src_x)
    direct    = torch.atan2(dy, dx)

    is_orbiting = (tgt_orb_r > 2.0)
    return torch.where(is_orbiting, intercept, direct)

class OrbitWarsEncoder(ObservationEncoder):
    def __init__(self, movement_horizon: int = 15):
        self.horizon = movement_horizon

    def encode(
        self, engine: TorchOrbitWarsEnv, player_per_env: Tensor
    ) -> DecisionBatch:
        B = engine.B
        device = engine.device

        inp = _encode_features(engine, player_per_env)

        # ── NEW: Orbital info for intercept angle ──
        dx0 = engine.planet_initial_x - CENTER
        dy0 = engine.planet_initial_y - CENTER
        orb_r     = torch.sqrt(dx0 * dx0 + dy0 * dy0)
        cur_angle = torch.atan2(
            engine.planet_y - CENTER,
            engine.planet_x - CENTER,
        )

        meta = {
            "planet_x":       engine.planet_x.clone(),
            "planet_y":       engine.planet_y.clone(),
            "planet_owner":   engine.planet_owner.clone(),
            "planet_ships":   engine.planet_ships.clone(),
            "planet_id":      engine.planet_id.clone(),
            "planet_alive":   (engine.planet_id >= 0),
            "player_per_env": player_per_env.clone(),
            "planet_orb_r":      orb_r.clone(),
            "planet_cur_angle":  cur_angle.clone(),
            "angular_velocity":  engine.angular_velocity.clone(),
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
        flow_matrix: torch.Tensor, # [B, P, P] from the policy
        num_envs: int,
    ) -> list[list[Move]]:
        
        MIN_GARRISON = 5  
        LAUNCH_THRESHOLD = 0.15 # Continuous Hold Gate filter
        
        meta  = batch.metadata
        px, py   = meta["planet_x"], meta["planet_y"]
        owner    = meta["planet_owner"]
        ships    = meta["planet_ships"]
        pids     = meta["planet_id"]
        alive    = meta["planet_alive"]
        seats    = meta["player_per_env"]
        orb_r    = meta["planet_orb_r"]
        cur_a    = meta["planet_cur_angle"]
        ang_vel  = meta["angular_velocity"]   
        B, P     = px.shape

        # Create valid mask grids [B, P, P]
        is_mine = (alive & (owner == seats.unsqueeze(1))).unsqueeze(2).expand(B, P, P)
        tgt_alive = alive.unsqueeze(1).expand(B, P, P)
        tgt_not_mine = (owner != seats.unsqueeze(1)).unsqueeze(1).expand(B, P, P)
        
        eye = torch.eye(P, device=px.device, dtype=torch.bool).unsqueeze(0).expand(B, P, P)

        # Condition: Value must break threshold, source is ours, target is valid, not self
        valid_launches = (flow_matrix > LAUNCH_THRESHOLD) & is_mine & tgt_alive & tgt_not_mine & ~eye

        moves: list[list[Move]] = [[] for _ in range(B)]
        
        # Scan active coordinates
        batch_indices, src_indices, tgt_indices = torch.nonzero(valid_launches, as_tuple=True)
        
        if batch_indices.numel() == 0:
            return moves

        for b, s, t in zip(batch_indices.tolist(), src_indices.tolist(), tgt_indices.tolist()):
            src_id = int(pids[b, s].item())
            src_ships = float(ships[b, s].item())
            
            available = max(src_ships - MIN_GARRISON, 0.0)
            n_send = int(math.floor(flow_matrix[b, s, t].item() * available))

            if n_send >= 1:
                sx, sy = px[b, s], py[b, s]
                tx, ty = px[b, t], py[b, t]
                
                # Orbit Intercept calculation
                dx, dy = tx - sx, ty - sy
                dist = math.sqrt(dx*dx + dy*dy)
                t_est = dist / 2.3
                
                t_orb = float(orb_r[b, t].item())
                if t_orb > 2.0:
                    t_angle = float(cur_a[b, t].item())
                    vel = float(ang_vel[b].item())
                    future_a = t_angle + vel * t_est
                    fx = 50.0 + t_orb * math.cos(future_a)
                    fy = 50.0 + t_orb * math.sin(future_a)
                    angle = math.atan2(fy - sy, fx - sx)
                else:
                    angle = math.atan2(dy, dx)

                moves[b].append([src_id, angle, n_send])
                
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


class SelfPlayOpponent(Opponent):
    """
    Mixes a frozen policy snapshot with the greedy fallback.
    Snapshot is synced every N training updates via sync_from().
    """
    def __init__(
        self,
        encoder:         OrbitWarsEncoder,
        device:          torch.device,
        greedy_mix_prob: float = 0.3,
        greedy_send_frac: float = 0.60,
    ):
        self.encoder         = encoder
        self.device          = device
        self.greedy_mix_prob = greedy_mix_prob
        self._greedy         = GreedyVectorOpponent(send_frac=greedy_send_frac)
        self._snapshot       = None   

    def sync_from(self, policy) -> None:
        """Copy current policy weights into the frozen snapshot."""
        if self._snapshot is None:
            self._snapshot = copy.deepcopy(policy)
            self._snapshot.eval()
        else:
            self._snapshot.load_state_dict(
                {k: v.clone() for k, v in policy.state_dict().items()}
            )

    def act(self, engine: TorchOrbitWarsEnv, seat_per_env: Tensor) -> list[list[Move]]:
        if self._snapshot is None:
            return self._greedy.act(engine, seat_per_env)

        use_greedy = torch.rand(engine.B) < self.greedy_mix_prob

        if use_greedy.all():
            return self._greedy.act(engine, seat_per_env)

        batch = self.encoder.encode(engine, seat_per_env)
        with torch.no_grad():
            out   = self._snapshot(batch)
            
        dist    = Categorical(logits=out.target_logits)
        actions = dist.sample()   # [B]
        policy_moves = self.encoder.decode_actions(batch, actions, engine.B)

        if not use_greedy.any():
            return policy_moves

        greedy_moves = self._greedy.act(engine, seat_per_env)
        return [
            greedy_moves[i] if use_greedy[i].item() else policy_moves[i]
            for i in range(engine.B)
        ]


__all__ = [
    "OrbitWarsEncoder",
    "MultiAgentOpponent",
    "GreedyVectorOpponent",
    "RLPolicyInput",
    "SelfPlayOpponent"
]