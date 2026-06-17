import os
import math
import torch

# Import your RL policy
from ppo_policy import OrbitWarsPolicy
from interfaces import DecisionBatch
from rl_bridge import RLPolicyInput

# Import parsing logic from your rule-based agent
from anotherAgent import single_obs_to_tensor, parse_obs, CENTER, ROT_RADIUS_LIMIT

# ---------------------------------------------------------
# 1. Global Model Initialization
# We load the model once when the script starts to avoid 
# loading overhead on every single turn (which causes timeouts).
# ---------------------------------------------------------
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# --- UPDATE THIS TO YOUR LATEST CHECKPOINT ---
CHECKPOINT_PATH = "/usershome/cs671_user11/OrbitWars/RL-Based/RL_infra/checkpoints/policy_6553600.pt" 

policy = OrbitWarsPolicy(feat_dim=12, d_model=64, n_heads=4, n_layers=2).to(device)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(checkpoint["policy_state"])
    policy.eval() # Set to evaluation mode (disables dropout)
else:
    print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Running with random weights.")

# We must cache initial_planets because the raw Kaggle environment 
# only provides current planet states on later turns.
_initial_planets_cache = None

def agent(obs):
    """
    The entry point for Kaggle / Local evaluation.
    Takes the raw `obs` dict, passes it through the RL policy, and returns moves.
    """
    global _initial_planets_cache
    
    # 1. Resolve basic state info
    player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    step = obs.get("step", 0) if isinstance(obs, dict) else obs.step
    
    # Cache initial planets on turn 0 for orbital geometry calculations
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    if step == 0 or _initial_planets_cache is None:
        _initial_planets_cache = [p.copy() if isinstance(p, list) else p for p in raw_planets]
        
    # Inject initial_planets into obs so anotherAgent's parser works
    if isinstance(obs, dict) and "initial_planets" not in obs:
        obs["initial_planets"] = _initial_planets_cache

    # 2. Convert raw obs to tensors
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id, device=device)
    
    # Add a dummy batch dimension (B=1) because the Network expects batches
    for k, v in obs_tensors.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            obs_tensors[k] = v.unsqueeze(0)
        elif isinstance(v, torch.Tensor): # Catch scalars
            obs_tensors[k] = v.unsqueeze(0)

    # 3. Extract exact RL features using your parser
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
    
    # Stack into [1, P, 12] exactly as trained
    features = torch.stack([
        is_mine, is_enemy, is_neutral,
        ships_norm, prod_norm, dist_sun_norm,
        torch.sin(cur_angle), torch.cos(cur_angle),
        torch.sin(orb_a0), torch.cos(orb_a0),
        parsed.is_orbiting.float(), parsed.alive.float()
    ], dim=-1)

    # 4. Wrap into a DecisionBatch
    inp = RLPolicyInput(features=features, alive_mask=parsed.alive)
    batch = DecisionBatch(
        policy_input=inp,
        env_index=torch.zeros(1, dtype=torch.long, device=device),
        rows_per_env=torch.ones(1, dtype=torch.long, device=device),
        metadata={}
    )

    # 5. Ask the Neural Network for the best target!
    with torch.no_grad():
        out = policy(batch)
        # Use argmax instead of sampling to pick the mathematically "best" move
        target_idx = out.target_logits.argmax(dim=-1).squeeze(0)

    # 6. Decode the target index into Kaggle's Move format
    moves = []
    
    my_mask = parsed.owned.squeeze(0) & parsed.alive.squeeze(0)
    if my_mask.any():
        my_ships = parsed.ships.squeeze(0).masked_fill(~my_mask, -1.0)
        best_src = my_ships.argmax() # Pick owned planet with the most ships
        
        # Extract coordinates and actual planet ID
        src_x = parsed.x.squeeze(0)[best_src]
        src_y = parsed.y.squeeze(0)[best_src]
        tgt_x = parsed.x.squeeze(0)[target_idx]
        tgt_y = parsed.y.squeeze(0)[target_idx]
        
        src_id = int(obs_tensors["planets"].squeeze(0)[best_src, 0].item())
        ships_avail = float(parsed.ships.squeeze(0)[best_src].item())
        
        # Send 55% of the garrison (mirroring how it was trained)
        n_send = int(math.floor(ships_avail * 0.55))
        
        # Don't attack ourselves, and only send if we have ships
        if best_src != target_idx and n_send > 0:
            angle = float(torch.atan2(tgt_y - src_y, tgt_x - src_x).item())
            moves.append([src_id, angle, n_send])

    return moves