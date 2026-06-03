import json
import glob
import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
REPLAY_DIR = "./aurora-replays"
MY_AGENT_NAME = "S4K50M"

def is_static(p):
    # Planet format: [id, owner, x, y, radius, ships, production]
    x, y, radius = p[2], p[3], p[4]
    dist_sq = (x - 50.0)**2 + (y - 50.0)**2
    return dist_sq >= (50.0 - radius)**2

def analyze_replays():
    replay_files = glob.glob(os.path.join(REPLAY_DIR, "*.json"))
    print(f"Found {len(replay_files)} replays. Analyzing...")

    # Data aggregators
    time_bleed = {"WIN": [], "LOSS": []}
    my_prod = {"WIN": [], "LOSS": []}
    enemy_prod = {"WIN": [], "LOSS": []}
    max_static_garrison = {"WIN": [], "LOSS": []}
    
    for filepath in replay_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # 1. Find S4K50M's player ID
        agents = data["info"].get("Agents", [])
        my_id = -1
        for idx, agent in enumerate(agents):
            if agent.get("Name") == MY_AGENT_NAME:
                my_id = idx
                break
                
        if my_id == -1:
            continue # Skipped, S4K50M not in this match
            
        # 2. Determine Win/Loss
        # rewards array corresponds to agent ids
        rewards = data.get("rewards", [0, 0])
        is_win = "WIN" if rewards[my_id] == 1 else "LOSS"
        
        # Initialize match tracking arrays
        match_time_bleed = []
        match_my_prod = []
        match_enemy_prod = []
        match_max_garrison = []
        
        steps = data.get("steps", [])
        for t, step in enumerate(steps):
            # Overage time check
            obs = step[0]["observation"]
            my_obs = step[my_id]["observation"]
            match_time_bleed.append(my_obs.get("remainingOverageTime", 60.0))
            
            # Economics & Garrisons
            planets = obs.get("planets", [])
            current_my_prod = 0
            current_enemy_prod = 0
            highest_static_ships = 0
            
            for p in planets:
                owner = p[1]
                ships = p[5]
                prod = p[6]
                
                if owner == my_id:
                    current_my_prod += prod
                    if is_static(p):
                        highest_static_ships = max(highest_static_ships, ships)
                elif owner != -1:
                    current_enemy_prod += prod
                    
            match_my_prod.append(current_my_prod)
            match_enemy_prod.append(current_enemy_prod)
            match_max_garrison.append(highest_static_ships)
            
        # Append to aggregators
        time_bleed[is_win].append(match_time_bleed)
        my_prod[is_win].append(match_my_prod)
        enemy_prod[is_win].append(match_enemy_prod)
        max_static_garrison[is_win].append(match_max_garrison)

    plot_insights(time_bleed, my_prod, enemy_prod, max_static_garrison)

def pad_arrays(arrays, max_len=500):
    # Pads arrays with their last value so we can average them easily
    padded = []
    for arr in arrays:
        if len(arr) == 0: continue
        arr_copy = arr.copy()
        while len(arr_copy) < max_len:
            arr_copy.append(arr_copy[-1])
        padded.append(arr_copy[:max_len])
    return padded

def get_average(arrays):
    padded = pad_arrays(arrays)
    if not padded: return [0]*500
    avg = []
    for t in range(500):
        val = sum(arr[t] for arr in padded) / len(padded)
        avg.append(val)
    return avg

def plot_insights(time_bleed, my_prod, enemy_prod, max_static_garrison):
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # --- PLOT 1: Time Bleed (The Timeout Check) ---
    axs[0].plot(get_average(time_bleed["WIN"]), label='Wins (Overage Time)', color='blue')
    axs[0].plot(get_average(time_bleed["LOSS"]), label='Losses (Overage Time)', color='red')
    axs[0].set_title('Average Remaining Compute Time (Overage) over 500 Ticks')
    axs[0].set_ylabel('Seconds')
    axs[0].axhline(y=0, color='black', linestyle='--')
    axs[0].legend()
    
    # --- PLOT 2: Economic Inflection (Macro Game) ---
    axs[1].plot(get_average(my_prod["WIN"]), label='My Prod (Wins)', color='blue')
    axs[1].plot(get_average(enemy_prod["WIN"]), label='Enemy Prod (Wins)', color='lightblue', linestyle='--')
    axs[1].plot(get_average(my_prod["LOSS"]), label='My Prod (Losses)', color='red')
    axs[1].plot(get_average(enemy_prod["LOSS"]), label='Enemy Prod (Losses)', color='salmon', linestyle='--')
    axs[1].set_title('Economic Trajectory (Production Mass)')
    axs[1].set_ylabel('Total Production / Tick')
    axs[1].legend()
    
    # --- PLOT 3: The Ghost Anvil Tracker (FSM Deadlock Check) ---
    axs[2].plot(get_average(max_static_garrison["WIN"]), label='Max Static Garrison (Wins)', color='blue')
    axs[2].plot(get_average(max_static_garrison["LOSS"]), label='Max Static Garrison (Losses)', color='red')
    axs[2].set_title('Anvil Formation: Max Ships on a Single Static Planet')
    axs[2].set_ylabel('Ship Count')
    axs[2].axhline(y=120, color='gray', linestyle='--', label='Mace Threshold (120)')
    axs[2].axhline(y=250, color='black', linestyle='--', label='Hammer Threshold (250)')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('aurora_forge_eda.png')
    print("Analysis complete. Check 'aurora_forge_eda.png'.")
    plt.show()

if __name__ == "__main__":
    analyze_replays()