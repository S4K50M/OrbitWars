import os
# --- CRITICAL: PREVENT CPU THREAD EXPLOSION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import math
import torch
# Force PyTorch to use 1 CPU thread per process
torch.set_num_threads(1)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

# Import your RL infrastructure
from engine import TorchEngineConfig
from vector_env import GpuVectorEnv
from rl_bridge import OrbitWarsEncoder, MultiAgentOpponent, GreedyVectorOpponent
from ppo_policy import OrbitWarsPolicy

import Experiments.agent as top_heuristic

# --- HYPERPARAMETERS ---
NUM_ENVS = 128        
MAX_PLANETS = 40
UPDATES = 1500         
WAIT_KEEP_PROB = 0.10  

def get_flow_labels(engine, teacher_moves):
    B, P = engine.B, engine.P
    device = engine.device

    flow_labels = torch.zeros(B, P, P, dtype=torch.float32, device=device)
    has_move = torch.zeros(B, dtype=torch.bool, device=device)

    px, py = engine.planet_x, engine.planet_y
    pids   = engine.planet_id
    ships  = engine.planet_ships

    for b in range(B):
        moves = teacher_moves[b]
        if not moves: continue
        has_move[b] = True

        for src_id, angle, n_ships in moves:
            src_match = (pids[b] == src_id).nonzero(as_tuple=True)[0]
            if len(src_match) == 0: continue
            p = src_match[0].item()

            sx, sy = px[b, p], py[b, p]
            diffs = (torch.atan2(py[b] - sy, px[b] - sx) - angle).abs()
            diffs = torch.where(diffs > math.pi, 2 * math.pi - diffs, diffs)
            diffs[p] = 999.0
            diffs.masked_fill_(pids[b] < 0, 999.0)
            
            best_diff, tgt_p = diffs.min(0)
            if best_diff.item() >= 0.5: continue  # drop noisy misaligned vectors

            available = max(ships[b, p].item() - 5.0, 1.0)
            # Encode exact fraction directly into the spatial coordinate coordinate matrix
            flow_labels[b, p, tgt_p.item()] = min(n_ships / available, 1.0)

    return flow_labels, has_move


def main():
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:4" if num_gpus > 4 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Starting Live Behavioral Cloning on {device}...")

    encoder = OrbitWarsEncoder(movement_horizon=15)
    engine_cfg = TorchEngineConfig(num_envs=NUM_ENVS, num_agents=2, device=device)
    
    env = GpuVectorEnv(
        engine_cfg=engine_cfg, 
        encoder=encoder, 
        opponent=GreedyVectorOpponent(), # Make the enemy greedy
        learner_seat=0
    )
    
    teacher = MultiAgentOpponent(agent_modules=[top_heuristic], max_workers=8) 
    policy = OrbitWarsPolicy(feat_dim=16, d_model=128, n_heads=4, n_layers=4).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    MAX_GRAD_NORM = 0.5  

    batch = env.reset()
    running_loss = 0.0
    trained_steps = 0   
    t0 = time.perf_counter()

    for update in range(1, UPDATES + 1):
        policy.train()
        optimizer.zero_grad()

        # 1. Forward Pass
        out = policy(batch)
        flow_pred = out.flow_matrix

        # 2. Get Teacher Labels
        teacher_moves = teacher.act(env.engine, env._learner_seat_tensor)
        flow_labels, has_move = get_flow_labels(env.engine, teacher_moves)

        # 3. Filter passivity
        keep_wait = torch.rand(NUM_ENVS, device=device) < WAIT_KEEP_PROB
        train_mask = has_move | keep_wait
        
        loss = 0.0
        if train_mask.any():
            # Filter active environments
            pred_slice = flow_pred[train_mask]
            label_slice = flow_labels[train_mask]

            # Mask valid actions to prevent scale suppression over P*P matrix
            inp = batch.policy_input
            src_ok = inp.is_mine_mask.unsqueeze(2).expand_as(flow_pred)
            tgt_ok = (inp.alive_mask & ~inp.is_mine_mask).unsqueeze(1).expand_as(flow_pred)
            eye = torch.eye(flow_pred.shape[1], device=device, dtype=torch.bool).unsqueeze(0)
            valid_mask = src_ok & tgt_ok & ~eye
            valid_slice = valid_mask[train_mask]

            # CRITICAL FIX: Scale weights to combat scale suppression
            # Active launches get a 15x gradient boost; passive cells get 1x
            loss_weights = torch.where(label_slice > 0.0, 15.0, 1.0)

            # Huber loss handles continuous regression gracefully without blowing up gradients
            base_loss = F.huber_loss(pred_slice, label_slice, delta=0.2, reduction="none")
            
            # Apply weights and valid mask
            masked_loss = (base_loss * loss_weights) * valid_slice
            
            valid_count = valid_slice.sum().clamp(min=1)
            loss = masked_loss.sum() / valid_count
            
            # Dummy add flow_logstd to loss so it receives gradients (keeps PyTorch distributed/optimizers happy)
            loss = loss + 0.0 * out.flow_logstd.sum()

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                running_loss += loss.item()
                trained_steps += 1

        # 4. Step Environment Forward using Teacher Actions
        result = env.step(batch, flow_labels) 
        batch = result.next_batch
        
        if update == 1: 
            print("Successfully completed first update. GPU/CPU Pipeline healthy.")

        if update % 50 == 0:
            avg_loss = running_loss / max(trained_steps, 1)
            elapsed = time.perf_counter() - t0
            sps = (50 * NUM_ENVS) / elapsed
            print(f"Update {update:4d}/{UPDATES} | Loss: {avg_loss:.4f} | SPS: {sps:.0f} | Time: {elapsed:.1f}s", flush=True)
            running_loss = 0.0
            trained_steps = 0
            t0 = time.perf_counter()
        
        # Periodic Checkpoint Saving
        if update % 300 == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            temp_path = f"checkpoints/bc_2p_baseline_up{update}.pt"
            torch.save({"policy_state": policy.state_dict()}, temp_path)
            print(f"--> Interim Checkpoint saved: {temp_path}", flush=True)
    Path("checkpoints").mkdir(exist_ok=True)
    save_path = "checkpoints/bc_2p_baseline.pt"
    torch.save({"policy_state": policy.state_dict()}, save_path)
    print(f"\nTraining Complete! Baseline saved to: {save_path}")

if __name__ == "__main__":
    main()