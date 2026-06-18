import os
import math
import torch

# Import your RL policy
from ppo_policy import OrbitWarsPolicy
from interfaces import DecisionBatch
from rl_bridge import RLPolicyInput

# Import parsing logic from your rule-based agent
from RuleAgent import single_obs_to_tensor, parse_obs, CENTER, ROT_RADIUS_LIMIT

# ---------------------------------------------------------
# Dynamic Device Selection
# Automatically routes to cuda:4 on the big rig, cuda:0 on 
# local laptops, or CPU if no GPU is found.
# ---------------------------------------------------------
if torch.cuda.is_available():
    if torch.cuda.device_count() > 4:
        device = torch.device("cuda:4")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# --- UPDATE THIS TO YOUR LATEST CHECKPOINT ---
CHECKPOINT_PATH = "/home/saksham/Documents/kaggle/OrbitWars/RL-Based/RL_infra/checkpoints/policy_72089600.pt" 

# Instantiate the upgraded V2 Architecture
policy = OrbitWarsPolicy(feat_dim=14, d_model=96, n_heads=4, n_layers=3).to(device)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["policy_state"]
    
    # Strip the '_orig_mod.' prefix added by torch.compile during training
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    policy.load_state_dict(clean_state_dict)
    policy.eval() # Set to evaluation mode (disables dropout)
else:
    print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Running with random weights.")

# Cache initial planets on turn 0 for orbital geometry calculations
_initial_planets_cache = None

def agent(obs):
    """
    The entry point for Kaggle / Local evaluation.
    Takes the raw `obs` dict, passes it through the RL policy, and returns moves.
    """
    global _initial_planets_cache
    
    player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    if step == 0 or _initial_planets_cache is None:
        _initial_planets_cache = [p.copy() if isinstance(p, list) else p for p in raw_planets]
        
    if isinstance(obs, dict) and "initial_planets" not in obs:
        obs["initial_planets"] = _initial_planets_cache

    # Convert raw obs to UNBATCHED tensors [P, 7]
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id, device=device)

    # Extract parsed features
    parsed = parse_obs(obs_tensors, player_id=player_id)
    
    is_mine = parsed.owned.float()
    is_enemy = parsed.is_enemy.float()
    is_neutral = parsed.is_neutral.float()
    ships_norm = (parsed.ships / 1000.0).clamp(max=1.0)
    prod_norm = (parsed.prod / 5.0).clamp(max=1.0)
    
    # Calculate angles natively
    dx0 = obs_tensors["initial_planets"][..., 2] - CENTER
    dy0 = obs_tensors["initial_planets"][..., 3] - CENTER
    orb_a0 = torch.atan2(dy0, dx0)
    
    dx = parsed.x - CENTER
    dy = parsed.y - CENTER
    cur_angle = torch.atan2(dy, dx)
    
    dist_sun_norm = (parsed.orb_r / float(ROT_RADIUS_LIMIT)).clamp(max=1.0)
    
    # --- NEW V2 FEATURES: Reachability & Proximity ---
    px = parsed.x
    py = parsed.y
    dx_all = px.unsqueeze(1) - px.unsqueeze(0)   # [P, P]
    dy_all = py.unsqueeze(1) - py.unsqueeze(0)
    dist_all = torch.sqrt((dx_all**2 + dy_all**2).clamp(min=0))  # [P, P]
    INF = 999.0
    
    owned_dist = dist_all.masked_fill(~is_mine.bool().unsqueeze(1).expand_as(dist_all), INF)
    dist_nearest_owned = owned_dist.amin(dim=0)  # [P] min dist from any owned planet
    
    reachable = (dist_nearest_owned < 35.0).float()
    dist_nearest_norm = (dist_nearest_owned / 70.0).clamp(max=1.0)

    # Stack into [P, 14]
    features_2d = torch.stack([
        is_mine, is_enemy, is_neutral,
        ships_norm, prod_norm, dist_sun_norm,
        torch.sin(cur_angle), torch.cos(cur_angle),
        torch.sin(orb_a0), torch.cos(orb_a0),
        parsed.is_orbiting.float(), parsed.alive.float(),
        reachable, dist_nearest_norm
    ], dim=-1)

    # Add the dummy batch dimension ONLY for the network -> [1, P, 14]
    features = features_2d.unsqueeze(0)
    alive_mask_batched = parsed.alive.unsqueeze(0)
    is_mine_mask_batched = parsed.owned.bool().unsqueeze(0)

    inp = RLPolicyInput(features=features, alive_mask=alive_mask_batched, is_mine_mask=is_mine_mask_batched)
    batch = DecisionBatch(
        policy_input=inp,
        env_index=torch.zeros(1, dtype=torch.long, device=device),
        rows_per_env=torch.ones(1, dtype=torch.long, device=device),
        metadata={}
    )

    # Ask the Neural Network for the best target!
    with torch.no_grad():
        out = policy(batch)
        flat_idx = out.target_logits.argmax(dim=-1).item()

    # --- NEW V2 DECODER: Unpack Bilinear flat index ---
    P = features_2d.shape[0]
    src_idx = flat_idx // P
    tgt_idx = flat_idx % P
    
    moves = []
    
    src_id = int(obs_tensors["planets"][src_idx, 0].item())
    src_ships = float(parsed.ships[src_idx].item())
    
    # Send 75% of the garrison (matching V2 training)
    n_send = int(math.floor(src_ships * 0.75))
    
    if src_idx != tgt_idx and n_send >= 5:
        src_x, src_y = parsed.x[src_idx], parsed.y[src_idx]
        tgt_x, tgt_y = parsed.x[tgt_idx], parsed.y[tgt_idx]
        angle = float(torch.atan2(tgt_y - src_y, tgt_x - src_x).item())
        moves.append([src_id, angle, n_send])

    return moves