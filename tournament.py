import os
import random
import time
from collections import defaultdict
from kaggle_environments import make
import itertools

# ============================================================
# TOURNAMENT CONFIGURATION
# ============================================================
AGENTS = [
    "./Testing-agents/apex-predator.py",
    "./Testing-agents/baiter-agent.py",
    "./Testing-agents/macro-king-agent.py",
    "./Testing-agents/rusher-agent.py",
    "./Testing-agents/sniper.py",
    "./Testing-agents/turtle-agent.py",
    "./Testing-agents/vulture-agent.py",
    "./agents/c-optimized-sniper.py",
    "./agents/b-Snipe+No-BOM.py",
    "./openAgents/fighter1.py"
]

SWISS_ROUNDS = 4
ITERATIONS_PER_MATCHUP = 5  # High iterations to iron out spawn variance
REPLAY_DIR = "./tourna-replays"

# Global Metrics Trackers
global_wins = defaultdict(int)
global_matches = defaultdict(int)
global_cpu_time = defaultdict(float)
global_steps_computed = defaultdict(int)

if not os.path.exists(REPLAY_DIR):
    os.makedirs(REPLAY_DIR)

# ============================================================
# CORE CORE ENVIRONMENT SIMULATION RUNNER
# ============================================================
def execute_match(match_agents, match_type, match_id, iteration):
    """
    Executes an Orbit Wars simulation matching the exact structure of your local setup.
    Tracks exact cumulative player durations and outputs inline live updates.
    """
    map_seed = random.randint(1, 999999)
    match_name = f"match_{match_id}_{match_type}_seed{map_seed}"
    
    # Mirroring your environment configuration exactly
    env = make("orbit_wars", configuration={"seed": map_seed}, debug=True)
    
    start_wall_time = time.time()
    env.run(match_agents)
    wall_duration = time.time() - start_wall_time
    
    num_players = len(match_agents)
    cumulative_agent_times = [0.0] * num_players
    
    # Exact step-by-step telemetry extraction from your script
    for step in env.steps[1:]:
        for player_idx, state in enumerate(step):
            duration = state.get('duration', 0.0) if isinstance(state, dict) else getattr(state, 'duration', 0.0)
            if duration is not None:
                cumulative_agent_times[player_idx] += duration

    # Extract final standings
    final_step = env.steps[-1]
    sorted_player_indices = sorted(
        range(num_players), 
        key=lambda idx: (final_step[idx].reward if final_step[idx].reward is not None else -1), 
        reverse=True
    )
    
    winner_idx = sorted_player_indices[0]
    winner_file = match_agents[winner_idx]
    
    # --- Live Result Double Logging (Requirement 1: Match Ended) ---
    print(f"\n[LIVE END] {match_type.upper()} | Match ID: {match_id} | Iteration: {iteration} | Seed: {map_seed}")
    print("-" * 75)
    for rank, idx in enumerate(sorted_player_indices, 1):
        agent_name = match_agents[idx].split('/')[-1]
        reward_display = f"{final_step[idx].reward:.1f}" if final_step[idx].reward is not None else "Elim."
        print(f" Rank {rank} | Player {idx} : {agent_name:<22} | Score: {reward_display:>6} | CPU: {cumulative_agent_times[idx]:.3f}s ({final_step[idx].status})")
    print("-" * 75)

    # Accumulate global profiling and leaderboard stats
    for idx, agent_path in enumerate(match_agents):
        global_matches[agent_path] += 1
        global_cpu_time[agent_path] += cumulative_agent_times[idx]
        global_steps_computed[agent_path] += (len(env.steps) - 1)
        if idx == winner_idx:
            global_wins[agent_path] += 1

    # Save Replay html
    html_output = env.render(mode="html", width=800, height=600)
    filename = f"{REPLAY_DIR}/replay_{match_name}_iter{iteration}.html"
    with open(filename, "w") as f:
        f.write(html_output)
        
    return winner_file

def print_leaderboard(title):
    """Prints a beautifully aligned standing sheet matching your layout."""
    print(f"\n=========================================================================")
    print(f"🏆 {title.upper()} 🏆")
    print(f"=========================================================================")
    print(f"{'Agent Name':<25} | {'Wins':<6} | {'Matches':<8} | {'Win Rate':<9} | {'Avg CPU/Step':<12}")
    print("-" * 75)
    
    summary = []
    for agent in AGENTS:
        wins = global_wins[agent]
        matches = global_matches[agent]
        win_rate = (wins / matches * 100) if matches > 0 else 0.0
        avg_cpu = (global_cpu_time[agent] / global_steps_computed[agent]) if global_steps_computed[agent] > 0 else 0.0
        summary.append((agent, wins, matches, win_rate, avg_cpu))
        
    # Rank by win rate descending
    summary.sort(key=lambda x: x[3], reverse=True)
    
    for rank, (agent, w, m, wr, cpu) in enumerate(summary, 1):
        clean_name = agent.split('/')[-1]
        print(f"{rank:>2}. {clean_name:<21} | {w:<6} | {m:<8} | {wr:>7.1f}% | {cpu:>10.4f}s")
    print("=========================================================================\n")

# ============================================================
# RUNNER PIPELINE EXECUTION
# ============================================================
if __name__ == "__main__":
    print("Initializing Multi-Scale Master Tournament Engine...")
    
    # --------------------------------------------------------
    # PHASE 1: Swiss 1v1 System (Gladiator Phase)
    # --------------------------------------------------------
    swiss_pool = list(AGENTS)
    match_counter = 1
    
    for round_idx in range(1, SWISS_ROUNDS + 1):
        print(f"\n>>> Commencing Swiss Bracket Round {round_idx} of {SWISS_ROUNDS}...")
        # Sort pool by current global wins to ensure parity matchmaking
        swiss_pool.sort(key=lambda x: global_wins[x], reverse=True)
        
        pairings = []
        temp_pool = list(swiss_pool)
        while len(temp_pool) >= 2:
            pairings.append((temp_pool.pop(0), temp_pool.pop(0)))
            
        for p1, p2 in pairings:
            for iter_idx in range(1, ITERATIONS_PER_MATCHUP + 1):
                execute_match([p1, p2], "swiss_1v1", match_counter, iter_idx)
            match_counter += 1

    # --------------------------------------------------------
    # PHASE 2: 4-Player FFA Groups (20 Unique Combinations)
    # --------------------------------------------------------
    print(f"\n=========================================================================")
    print(f"STAGE 2: 4-PLAYER FFA (CHAOS ARENA - 20 UNIQUE MATCHES)")
    print(f"=========================================================================")
    
    # Sort seeds cleanly by global win rate from Stage 1
    ranked_seeds = sorted(AGENTS, key=lambda x: (global_wins[x] / global_matches[x] if global_matches[x] > 0 else 0), reverse=True)
    
    # 4 distinct groups of 5. Each agent appears exactly twice.
    groups = {
        "Group_A_Snake": [ranked_seeds[0], ranked_seeds[3], ranked_seeds[4], ranked_seeds[7], ranked_seeds[8]],
        "Group_B_Snake": [ranked_seeds[1], ranked_seeds[2], ranked_seeds[5], ranked_seeds[6], ranked_seeds[9]],
        "Group_C_Top5":  [ranked_seeds[0], ranked_seeds[1], ranked_seeds[2], ranked_seeds[3], ranked_seeds[4]],
        "Group_D_Bot5":  [ranked_seeds[5], ranked_seeds[6], ranked_seeds[7], ranked_seeds[8], ranked_seeds[9]]
    }
    
    match_counter = 1
    for g_name, g_agents in groups.items():
        print(f"\n>>> Launching Battle Royale Arena: {g_name}...")
        
        # itertools.combinations(list, 4) mathematically yields exactly 5 unique subsets of 4 agents
        unique_matchups = list(itertools.combinations(g_agents, 4))
        
        for match_agents in unique_matchups:
            # Convert tuple to list and shuffle to randomize spawn positions (P1 vs P4 advantage)
            shuffled = list(match_agents)
            random.shuffle(shuffled)
            
            # Execute exactly 1 iteration of this specific unique combination
            execute_match(shuffled, "ffa_4way", f"{g_name}_M{match_counter}", 1)
            match_counter += 1
    # --------------------------------------------------------
    # PHASE 3: The Grand Finals Arena
    # --------------------------------------------------------
    final_ranked = sorted(AGENTS, key=lambda x: (global_wins[x] / global_matches[x] if global_matches[x] > 0 else 0), reverse=True)
    grand_finalists = final_ranked[:4]
    
    print(f"\n=========================================================================")
    print(f"🔥 UNLEASHING THE GRAND FINALS ARENA 🔥")
    print(f"Qualifying Competitors: {[f.split('/')[-1] for f in grand_finalists]}")
    print(f"=========================================================================")
    
    for iter_idx in range(1, 11): # 10 final master iterations
        final_shuffled = list(grand_finalists)
        random.shuffle(final_shuffled)
        execute_match(final_shuffled, "grand_final", "CHAMPIONSHIP", iter_idx)

    # --- Final Result Aggregation (Requirement 2: In The End Summary) ---
    print_leaderboard("Final Tournament Matrix Results")