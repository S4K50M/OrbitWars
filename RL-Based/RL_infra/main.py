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
CHECKPOINT_PATH = "/home/saksham/Documents/kaggle/OrbitWars/RL-Based/RL_infra/checkpoints/bc_2p_baseline_up600.pt" 

# Instantiate the upgraded V2 Architecture
policy = OrbitWarsPolicy(feat_dim=16, d_model=128, n_heads=4, n_layers=4).to(device)

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
    
    # ── V2 FEATURES: Reachability & Proximity ──
    px = parsed.x
    py = parsed.y
    dx_all = px.unsqueeze(1) - px.unsqueeze(0)   # [P, P]
    dy_all = py.unsqueeze(1) - py.unsqueeze(0)
    dist_all = torch.sqrt((dx_all**2 + dy_all**2).clamp(min=0))  # [P, P]
    INF = 999.0
    
    owned_dist = dist_all.masked_fill(~is_mine.bool().unsqueeze(1).expand_as(dist_all), INF)
    dist_nearest_owned = owned_dist.amin(dim=0)  # [P]
    
    reachable = (dist_nearest_owned < 35.0).float()
    dist_nearest_norm = (dist_nearest_owned / 70.0).clamp(max=1.0)

    # ── V3 FEATURES: Global Game Phase & Economy ──
    P = parsed.x.shape[0]
    step_norm = torch.tensor([step / 500.0], device=device).expand(P)
    
    prod = parsed.prod
    total_prod = prod.sum().clamp(min=1.0)
    my_prod = (prod * is_mine).sum()
    prod_ratio = (my_prod / total_prod).expand(P)

    # Stack into [P, 16]
    features_2d = torch.stack([
        is_mine, is_enemy, is_neutral,
        ships_norm, prod_norm, dist_sun_norm,
        torch.sin(cur_angle), torch.cos(cur_angle),
        torch.sin(orb_a0), torch.cos(orb_a0),
        parsed.is_orbiting.float(), parsed.alive.float(),
        reachable, dist_nearest_norm,
        step_norm, prod_ratio
    ], dim=-1)

    # Add the dummy batch dimension ONLY for the network -> [1, P, 16]
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

    # ── NEW MULTI-HEAD EVALUATION ──
    with torch.no_grad():
        out = policy(batch)
        
        # Deterministic Evaluation (No sampling noise for actual Kaggle play)
        # Gate > 0 means the network is confident this planet should fire
        gate_actions = (out.gate_logits[0] > 0) & is_mine_mask_batched[0]
        tgt_actions  = out.tgt_logits[0].argmax(dim=-1)
        ship_frac    = out.ship_frac[0]

    moves = []
    MIN_GARRISON = 5

    # Pull variables to CPU for standard Python math operations
    gate_cpu = gate_actions.cpu().tolist()
    tgt_cpu  = tgt_actions.cpu().tolist()
    frac_cpu = ship_frac.cpu().tolist()

    # Iterate over every planet on the board
    for p in range(P):
        if gate_cpu[p]:  # If the network's launch gate fired for this specific planet
            tgt = tgt_cpu[p]
            
            src_ships = float(parsed.ships[p].item())
            # Ensure we always leave a minimum garrison behind
            available = max(src_ships - MIN_GARRISON, 0.0)
            n_send = int(math.floor(frac_cpu[p] * available))
            
            # If we aren't targeting ourselves and have ships to send
            if tgt != p and n_send >= 1:
                src_x, src_y = parsed.x[p].item(), parsed.y[p].item()
                tgt_x, tgt_y = parsed.x[tgt].item(), parsed.y[tgt].item()
                
                # Intercept Geometry 
                dx = tgt_x - src_x
                dy = tgt_y - src_y
                dist = math.sqrt(dx**2 + dy**2)
                t_est = dist / 2.3  # Approximate Fleet Speed
                
                tgt_orb_r = float(parsed.orb_r[tgt].item())
                
                if tgt_orb_r > 2.0:
                    # Moving target: Lead the intercept angle
                    tgt_cur_a = math.atan2(tgt_y - CENTER, tgt_x - CENTER)
                    ang_vel = float(obs_tensors["angular_velocity"].item())
                    future_a = tgt_cur_a + ang_vel * t_est
                    future_x = CENTER + tgt_orb_r * math.cos(future_a)
                    future_y = CENTER + tgt_orb_r * math.sin(future_a)
                    angle = math.atan2(future_y - src_y, future_x - src_x)
                else:
                    # Static target (e.g. sun/center clusters if applicable)
                    angle = math.atan2(dy, dx)
                
                src_id = int(obs_tensors["planets"][p, 0].item())
                moves.append([src_id, angle, n_send])

    return moves