import json
import glob
import os
import math
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
REPLAY_DIR = "./aurora-replays"  # Scans the current working folder for .json files
MY_AGENT_NAME = "S4K50M"
MAX_TICKS = 500

def analyze_advanced_telemetry():
    replay_files = glob.glob(os.path.join(REPLAY_DIR, "*.json"))
    if not replay_files:
        print(f"[ERROR] No JSON files found in directory '{REPLAY_DIR}'")
        return
        
    print(f"Found {len(replay_files)} JSON files. Initiating Spatial Threat Analysis...\n")

    modes = {
        "1v1": {"WIN": [], "LOSS": []},
        "FFA": {"WIN": [], "LOSS": []}
    }
    
    stats_summary = {"1v1_wins": 0, "1v1_losses": 0, "FFA_wins": 0, "FFA_losses": 0}

    for filepath in replay_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Unwrap if Kaggle API wrapped it in a single-element array
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    data = data[0]
                else:
                    continue
                
            if not isinstance(data, dict) or "steps" not in data:
                print(f"[SKIP] {filename} -> Invalid schema: Missing 'steps' array")
                continue
                
            # --- 1. Aggressive ID Discovery ---
            info = data.get("info", {})
            team_names = info.get("TeamNames", [])
            agents = info.get("Agents", [])
            
            my_id = -1
            num_players = 0
            
            if team_names and MY_AGENT_NAME in team_names:
                my_id = team_names.index(MY_AGENT_NAME)
                num_players = len(team_names)
            elif agents:
                num_players = len(agents)
                for idx, ag in enumerate(agents):
                    if isinstance(ag, dict) and ag.get("Name") == MY_AGENT_NAME:
                        my_id = idx
                        break
                        
            if my_id == -1:
                continue
                
            # Classify game type
            if num_players == 2:
                mode_key = "1v1"
            elif num_players > 2:
                mode_key = "FFA"
            else:
                continue

            # --- 2. Corrected Win/Loss Discovery ---
            rewards = data.get("rewards")
            if not rewards or len(rewards) <= my_id or rewards[my_id] is None:
                last_step = data["steps"][-1]
                rewards = [p.get("reward", -999) if p.get("reward") is not None else -999 for p in last_step]
            
            my_reward = rewards[my_id]
            
            if my_reward == max(rewards) and my_reward > -1:
                is_win = "WIN"
                summary_key = f"{mode_key}_wins"
            else:
                is_win = "LOSS"
                summary_key = f"{mode_key}_losses"
            
            stats_summary[summary_key] += 1
            
            # --- 3. Spatial Threat Extraction ---
            match_engagement = []   
            match_dispersion = []
            match_anvil_threat = []
            
            for t, step in enumerate(data.get("steps", [])):
                if not step or len(step) <= my_id: 
                    continue
                
                obs = step[0].get("observation", {})
                planets = obs.get("planets", [])
                fleets = obs.get("fleets", [])  
                
                my_planet_ships = 0
                total_my_ships = 0
                my_planet_coords = []
                
                # A. Base Metrics
                for p in planets:
                    owner, ships = p[1], p[5]
                    if owner == my_id:
                        my_planet_ships += ships
                        total_my_ships += ships
                        my_planet_coords.append((p[2], p[3]))
                
                my_fleet_ships = 0
                for f in fleets:
                    if len(f) >= 7 and f[6] == my_id:
                        my_fleet_ships += f[1]
                        total_my_ships += f[1]
                
                uptime = (my_fleet_ships / total_my_ships) if total_my_ships > 0 else 0
                match_engagement.append(uptime)
                
                # B. Center of Mass & Dispersion
                if len(my_planet_coords) > 0:
                    cx = sum(c[0] for c in my_planet_coords) / len(my_planet_coords)
                    cy = sum(c[1] for c in my_planet_coords) / len(my_planet_coords)
                else:
                    cx, cy = 50.0, 50.0 # Fallback map center if temporarily wiped out
                    
                if len(my_planet_coords) > 1:
                    avg_dist = sum(math.sqrt((c[0]-cx)**2 + (c[1]-cy)**2) for c in my_planet_coords) / len(my_planet_coords)
                    match_dispersion.append(avg_dist)
                else:
                    match_dispersion.append(0)
                    
                # C. Vector Threat Classification
                anvil_threat = 0
                
                for f in fleets:
                    if len(f) >= 7 and f[1] != my_id and f[1] != -1: 
                        f_x, f_y, f_angle, f_ships = f[2], f[3], f[4], f[6]
                        
                        dist_to_core = math.sqrt((f_x - cx)**2 + (f_y - cy)**2)
                        angle_to_core = math.atan2(cy - f_y, cx - f_x)
                        
                        # Normalize angle diff to handle pi/-pi wrap around
                        angle_diff = (f_angle - angle_to_core + math.pi) % (2 * math.pi) - math.pi
                        
                        # Target Cone: ~28 degrees threshold
                        if abs(angle_diff) < 0.5:
                            if dist_to_core < 15.0:
                                anvil_threat += f_ships

                match_anvil_threat.append(anvil_threat)
                    
            modes[mode_key][is_win].append({
                "engagement": match_engagement,
                "dispersion": match_dispersion,
                "anvil_threat": match_anvil_threat
            })
            
            print(f"[OK] Parsed {filename} -> {mode_key} {is_win}")
            
        except Exception as e:
            print(f"[ERROR] {filename} -> Parse crash: {str(e)}")

    generate_plots(modes)

def process_metric_average(match_list, key):
    if not match_list:
        return [0] * MAX_TICKS
    
    padded_matrix = []
    for match in match_list:
        arr = match[key]
        if not arr: 
            continue
        arr_copy = list(arr)
        while len(arr_copy) < MAX_TICKS:
            arr_copy.append(arr_copy[-1] if arr_copy else 0)
        padded_matrix.append(arr_copy[:MAX_TICKS])
        
    if not padded_matrix:
        return [0] * MAX_TICKS
    return np.mean(padded_matrix, axis=0)

def generate_plots(modes):
    # Upgraded to 3x2 grid to fit the Threat Analytics
    fig, axs = plt.subplots(3, 2, figsize=(16, 15))
    
    metrics_meta = [
        ("1v1", "engagement", 0, 0, "1v1: Airborne Force Multiplier", "Fraction of Force in Transit", "blue", "red"),
        ("FFA", "engagement", 0, 1, "FFA: Airborne Force Multiplier", "Fraction of Force in Transit", "darkgreen", "orange"),
        ("1v1", "dispersion", 1, 0, "1v1: Empire Cohesion Boundary", "Mean Units From Center of Mass", "blue", "red"),
        ("FFA", "dispersion", 1, 1, "FFA: Empire Cohesion Boundary", "Mean Units From Center of Mass", "darkgreen", "orange"),
        ("1v1", "anvil_threat", 2, 0, "1v1: Core Penetration (Anvil Threat)", "Incoming Enemy Ships (<15u)", "blue", "red"),
        ("FFA", "anvil_threat", 2, 1, "FFA: Core Penetration (Anvil Threat)", "Incoming Enemy Ships (<15u)", "darkgreen", "orange")
    ]
    
    for mode, key, r, c, title, ylabel, c1, c2 in metrics_meta:
        win_data = process_metric_average(modes[mode]["WIN"], key)
        loss_data = process_metric_average(modes[mode]["LOSS"], key)
        
        axs[r, c].plot(win_data, label=f"{mode} Wins", color=c1, linewidth=2.5)
        axs[r, c].plot(loss_data, label=f"{mode} Losses", color=c2, linewidth=2.5, linestyle="--" if mode=="FFA" else "-")
        axs[r, c].set_title(title, fontsize=12, fontweight='bold')
        axs[r, c].set_ylabel(ylabel)
        axs[r, c].set_xlabel("Ticks")
        axs[r, c].legend(loc="upper left")
        axs[r, c].grid(True, alpha=0.25)
    
    plt.tight_layout()
    output_filename = "threat_vector_telemetry.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nThreat Vector chart safely written to: '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    analyze_advanced_telemetry()