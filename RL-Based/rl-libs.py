import torch
from torch import Tensor
from dataclasses import dataclass

from orbit_lite.constants import *
from orbit_lite.obs import parse_obs

from orbit_lite.adapter import PlanetMovement, PlanetGarrisonStatus
from orbit_lite.movement_step import  LaunchEntries, _empty_entries, infer_planned_launches_from_entries, apply_private_planned_launches
from orbit_lite.planner_core import safe_drain,entries_to_sparse_payload
from orbit_lite.config import ProducerLiteConfig, ProducerLiteMemory, _movement_config

@dataclass
class OrbitWarsObsRL:
    """Normalized tensor payload for the neural network."""
    planet_features: Tensor  # [B, P, 12]
    planet_alive: Tensor     # [B, P] bool mask for transformer attention
    step_norm: Tensor        # [B, 1] 

def extract_rl_features(obs_tensors: dict, player_id: int) -> OrbitWarsObsRL:
    """
    Transforms the raw engine obs_tensors into normalized NN inputs.
    Fully leverages the existing parse_obs logic.
    """
    # parse_obs already uses `...` indexing, so it natively handles [B, P, 7]
    parsed = parse_obs(obs_tensors, player_id=player_id)
    
    B = parsed.alive.shape[0] if parsed.alive.dim() > 1 else 1
    P = parsed.P
    device = parsed.device
    dtype = torch.float32

    # 1. Ownership (One-Hot)
    is_mine = parsed.owned.to(dtype)
    is_enemy = parsed.is_enemy.to(dtype)
    is_neutral = parsed.is_neutral.to(dtype)

    # 2. Normalized Scale Features
    # Assuming max ships per planet rarely exceeds 1000, and max prod is 5
    ships_norm = (parsed.ships / 1000.0).clamp(max=1.0).to(dtype)
    prod_norm = (parsed.prod / 5.0).clamp(max=1.0).to(dtype)
    
    # 3. Orbital Features
    dist_to_sun_norm = (parsed.orb_r / ROT_RADIUS_LIMIT).clamp(max=1.0).to(dtype)
    is_orbiting = parsed.is_orbiting.to(dtype)
    
    # Calculate current angle from x, y
    dx = parsed.x - CENTER
    dy = parsed.y - CENTER
    current_angle = torch.atan2(dy, dx)
    
    cur_angle_sin = torch.sin(current_angle).to(dtype)
    cur_angle_cos = torch.cos(current_angle).to(dtype)
    init_angle_sin = torch.sin(parsed.orb_a0).to(dtype)
    init_angle_cos = torch.cos(parsed.orb_a0).to(dtype)
    
    alive_mask = parsed.alive.to(dtype)

    # Stack into [B, P, 12]
    # If unbatched [P], unsqueeze to [1, P] for stacking, then view as [B, P, 12]
    features = [
        is_mine, is_enemy, is_neutral, 
        ships_norm, prod_norm, dist_to_sun_norm,
        cur_angle_sin, cur_angle_cos, 
        init_angle_sin, init_angle_cos,
        is_orbiting, alive_mask
    ]
    
    # Reshape all to [B, P, 1] and concatenate
    features_reshaped = [f.view(B, P, 1) for f in features]
    planet_features = torch.cat(features_reshaped, dim=-1)

    step_norm = (parsed.step / float(DEFAULT_EPISODE_STEPS)).view(B, 1).to(dtype)

    return OrbitWarsObsRL(
        planet_features=planet_features,
        planet_alive=parsed.alive.view(B, P),
        step_norm=step_norm
    )

def policy_actions_to_entries(
    obs_tensors: dict,
    movement: PlanetMovement,
    garrison_status: PlanetGarrisonStatus,
    player_id: int,
    launch_logits: Tensor,     # [B, P] from Network
    angle_sin: Tensor,         # [B, P] from Network
    angle_cos: Tensor,         # [B, P] from Network
    ship_fractions: Tensor,    # [B, P] from Network (0.0 to 1.0)
    deterministic: bool = False
) -> LaunchEntries:
    """
    Translates RL policy outputs into the engine's LaunchEntries format.
    Uses safe_drain to ensure the RL agent cannot command illegal moves.
    """
    parsed = parse_obs(obs_tensors, player_id=player_id)
    device = parsed.device
    P = parsed.P
    
    # 1. Decide which planets are launching
    if deterministic:
        launch_mask = launch_logits > 0.0
    else:
        probs = torch.sigmoid(launch_logits)
        launch_mask = torch.bernoulli(probs).to(torch.bool)
        
    # Only allow launches from owned, alive planets
    valid_sources = launch_mask & parsed.owned & parsed.alive
    
    # 2. Extract Action Parameters for valid sources
    src_idx = torch.nonzero(valid_sources, as_tuple=True)[0] # Assuming B=1 for now, or use flattened indices
    
    if src_idx.numel() == 0:
        return _empty_entries(device, parsed.ships.dtype)
        
    fraction = ship_fractions[valid_sources].clamp(0.0, 1.0)
    sin_vals = angle_sin[valid_sources]
    cos_vals = angle_cos[valid_sources]
    angle = torch.atan2(sin_vals, cos_vals)
    
    # 3. Calculate allowed ships using existing safe_drain
    # The network decides the *fraction* of safe ships, the physics engine enforces the absolute ceiling
    H_eff = torch.full((), float(movement.movement_horizon), dtype=parsed.ships.dtype, device=device)
    source_ships = parsed.ships[src_idx]
    
    drain = safe_drain(
        garrison_status, 
        source_idx=src_idx, 
        source_ships=source_ships, 
        H_eff=H_eff, 
        player_id=player_id
    )
    
    # Calculate exact ship count
    ships_to_send = torch.floor(drain * fraction)
    valid_send = ships_to_send >= 1.0
    
    # 4. Dummy target slots (Engine requires a target_slot in LaunchEntries, 
    # but the actual sparse action payload only uses angle and from_planet_id)
    dummy_targets = torch.zeros_like(src_idx)
    dummy_etas = torch.ones_like(ships_to_send)
    
    entries = LaunchEntries(
        source_slots=src_idx,
        target_slots=dummy_targets, 
        ships=ships_to_send,
        angle=angle,
        eta=dummy_etas,
        valid=valid_send,
    )
    
    return entries


class BatchedOrbitWarsEnv:
    """
    Wraps the existing rule-based infrastructure into an RL-friendly environment.
    """
    def __init__(self, config: ProducerLiteConfig, player_id: int, player_count: int = 4):
        self.config = config
        self.player_id = player_id
        self.player_count = player_count
        self.memory = ProducerLiteMemory()
        
    def reset(self, initial_obs_tensors: dict) -> OrbitWarsObsRL:
        """
        Ingests a new game state (e.g., from kaggle episodes).
        """
        self.memory.reset()
        self.memory.cached_player_count = self.player_count
        
        # Initialize Movement Cache
        self.memory.movement = PlanetMovement.from_obs_tensors(
            initial_obs_tensors, 
            config=_movement_config(self.config, player_count=self.player_count)
        )
        
        return extract_rl_features(initial_obs_tensors, self.player_id)
        
    def step(self, obs_tensors: dict, policy_actions: dict) -> tuple[OrbitWarsObsRL, dict[str, Tensor]]:
        """
        Executes a single step using the Neural Network's actions.
        Returns the next observation features and the sparse payload for the game engine.
        """
        movement = self.memory.movement
        movement.update(obs_tensors)
        
        H = int(self.config.horizon)
        garrison_status = movement.garrison_status(max_horizon=H)
        
        # 1. Translate Network Actions -> LaunchEntries
        entries = policy_actions_to_entries(
            obs_tensors=obs_tensors,
            movement=movement,
            garrison_status=garrison_status,
            player_id=self.player_id,
            launch_logits=policy_actions['launch_logits'],
            angle_sin=policy_actions['angle_sin'],
            angle_cos=policy_actions['angle_cos'],
            ship_fractions=policy_actions['ship_fractions']
        )
        
        # 2. Let the exact physics engine assign IDs and track the fleets
        launches = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors, 
            movement=movement, 
            entries=entries, 
            player_id=self.player_id
        )
        
        apply_private_planned_launches(
            movement=movement, 
            launches=launches, 
            owner_id=self.player_id,
            obs_tensors=obs_tensors
        )
        
        # 3. Format payload for the Kaggle engine
        planet_ids = obs_tensors["planets"][..., 0].long()
        sparse_payload = entries_to_sparse_payload(entries, planet_ids=planet_ids)
        
        # 4. Extract features for the NEXT step
        next_obs_features = extract_rl_features(obs_tensors, self.player_id)
        
        return next_obs_features, sparse_payload

