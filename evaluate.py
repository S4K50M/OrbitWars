import os
import random
import time
from collections import defaultdict
from kaggle_environments import make

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Define our competitors - Update these paths as needed
agent_baseline = "/home/saksham/Documents/kaggle/OrbitWars/Experiments/MultiAgent.py"
agent_test = "/home/saksham/Documents/kaggle/OrbitWars/Experiments/LyonelHybrid.py"

# Number of unique map seeds to generate
NUM_SEEDS = 15  # This will result in 10x 1v1s and 10x 4v4s due to swapping

# Ensure replay directory exists
os.makedirs("./replays", exist_ok=True)

# Generate random seeds
seeds = [random.randint(1, 999999) for _ in range(NUM_SEEDS)]

# ---------------------------------------------------------
# Match Generation (Positional Swapping)
# ---------------------------------------------------------
matches = []
for seed in seeds:
    # 1v1: Swapped positions for absolute fairness
    matches.append({"mode": "1v1", "seed": seed, "agents": [agent_baseline, agent_test], "types": ["Baseline", "Test"]})
    # matches.append({"mode": "1v1", "seed": seed, "agents": [agent_test, agent_baseline], "types": ["Test", "Baseline"]})
    
    # 4v4: Swapped positions for FFA
    matches.append({"mode": "4v4", "seed": seed, "agents": [agent_baseline, agent_test, agent_baseline, agent_test], "types": ["Baseline", "Test", "Baseline", "Test"]})
    # matches.append({"mode": "4v4", "seed": seed, "agents": [agent_test, agent_baseline, agent_test, agent_baseline], "types": ["Test", "Baseline", "Test", "Baseline"]})

# ---------------------------------------------------------
# Tracking & Statistics
# ---------------------------------------------------------
results_summary = []
stats = {
    "1v1": {"test_wins": 0, "baseline_wins": 0, "ties": 0, "test_time": [], "base_time": []},
    "4v4": {"test_wins": 0, "baseline_wins": 0, "ties": 0, "test_time": [], "base_time": []}
}

print(f"Starting Strict Ablation Tournament: {len(matches)} Total Matches...\n")

for i, match in enumerate(matches, 1):
    mode = match['mode']
    print(f"[{i}/{len(matches)}] Running {mode} | Map Seed: {match['seed']} | Setup: {' vs '.join(match['types'])}")
    
    env = make("orbit_wars", configuration={"seed": match['seed']}, debug=True)
    start_time = time.time()
    
    # Run simulation
    env.run(match['agents'])
    sim_duration = time.time() - start_time

    # Calculate exact cumulative processing time
    num_players = len(match['agents'])
    cumulative_agent_times = [0.0] * num_players
    
    for step in env.steps[1:]:
        for player_idx, state in enumerate(step):
            duration = state.get('duration', 0.0) if isinstance(state, dict) else getattr(state, 'duration', 0.0)
            if duration is not None:
                cumulative_agent_times[player_idx] += duration

    # Extract results
    final_step = env.steps[-1]
    match_results = []
    
    for player_idx, state in enumerate(final_step):
        a_type = match['types'][player_idx]
        reward = state.reward if state.reward is not None else -float('inf')
        total_time = cumulative_agent_times[player_idx]
        
        match_results.append({
            "player_id": player_idx,
            "type": a_type,
            "reward": reward,
            "status": state.status,
            "total_time": total_time
        })
        
        # Track CPU times
        if a_type == "Test":
            stats[mode]["test_time"].append(total_time)
        else:
            stats[mode]["base_time"].append(total_time)

    # Determine Winner
    max_reward = max(r['reward'] for r in match_results)
    winners = [r for r in match_results if r['reward'] == max_reward]
    
    # Scoring Logic (1 for top agent, 0 for others)
    winner_types = [w['type'] for w in winners]
    if "Test" in winner_types and "Baseline" not in winner_types:
        stats[mode]["test_wins"] += 1
        winner_str = "🏆 TEST WON"
    elif "Baseline" in winner_types and "Test" not in winner_types:
        stats[mode]["baseline_wins"] += 1
        winner_str = "🛑 BASELINE WON"
    else:
        stats[mode]["ties"] += 1
        winner_str = "🤝 TIE"

    print(f"  -> {winner_str} (Sim time: {sim_duration:.2f}s)")
    
    # Save Replay
    html_output = env.render(mode="html", width=800, height=600)
    filename = f"./replays/match_{i}_{mode}_seed{match['seed']}.html"
    with open(filename, "w") as f:
        f.write(html_output)

# ---------------------------------------------------------
# Final Tournament Report
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("📊 ABLATION EVALUATION REPORT")
print("=" * 60)

for mode in ["1v1", "4v4"]:
    s = stats[mode]
    total_matches = s["test_wins"] + s["baseline_wins"] + s["ties"]
    
    if total_matches == 0:
        continue
        
    win_rate = (s["test_wins"] / total_matches) * 100
    
    avg_test_cpu = sum(s["test_time"]) / max(1, len(s["test_time"]))
    avg_base_cpu = sum(s["base_time"]) / max(1, len(s["base_time"]))
    
    print(f"\n[{mode} Results]")
    print(f"Total Matches : {total_matches}")
    print(f"Test Wins     : {s['test_wins']}")
    print(f"Baseline Wins : {s['baseline_wins']}")
    print(f"Ties          : {s['ties']}")
    print(f"WIN RATE      : {win_rate:.1f}%")
    print(f"Avg CPU Time  : Test = {avg_test_cpu:.2f}s | Baseline = {avg_base_cpu:.2f}s")
    
    if win_rate > 55.0:
        print("💡 Verdict     : CONFIRMED IMPROVEMENT. Deploy the change.")
    elif win_rate < 45.0:
        print("⚠️ Verdict     : DEGRADATION. Reject the change.")
    else:
        print("⚖️ Verdict      : INCONCLUSIVE. Run more seeds to break statistical noise.")

print("\nEvaluation complete. Replays saved to ./replays/")