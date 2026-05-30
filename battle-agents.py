import random
import time
from kaggle_environments import make

# Define our competitors
agent_a = "./agents/c-optimized-sniper.py"
agent_b = "./agents/b-Snipe+No-BOM.py"

# Generate 5 random seeds for the tournament
seeds = [random.randint(1, 999999) for _ in range(5)]

# Configure the 5 matches: 3x 2-player, 2x 4-player 
matches = [
    {"mode": "2-player", "seed": seeds[0], "agents": [agent_a, agent_b]},
    {"mode": "4-player", "seed": seeds[1], "agents": [agent_a, agent_b, agent_a, agent_b]},
    {"mode": "2-player", "seed": seeds[2], "agents": [agent_a, agent_b]},
    {"mode": "4-player", "seed": seeds[3], "agents": [agent_a, agent_b, agent_a, agent_b]},
    {"mode": "2-player", "seed": seeds[4], "agents": [agent_a, agent_b]}
]

results_summary = []

print("Starting Randomized Orbit Wars Local Tournament...\n")

for i, match in enumerate(matches, 1):
    print(f"Running Match {i} ({match['mode']}, Map Seed: {match['seed']})...")
    
    # Initialize environment with the random seed
    env = make("orbit_wars", configuration={"seed": match['seed']}, debug=True)
    
    # Track the total wall-clock time for the simulation
    start_time = time.time()
    
    # Run the match
    env.run(match['agents'])
    
    sim_duration = time.time() - start_time
    print(f"  -> Simulation completed in {sim_duration:.2f} seconds.")

    # Calculate exact cumulative processing time for each agent
    num_players = len(match['agents'])
    cumulative_agent_times = [0.0] * num_players
    
    # Start from step 1 because step 0 is just the initial environment observation
    for step in env.steps[1:]:
        for player_idx, state in enumerate(step):
            # Kaggle stores execution time in the 'duration' attribute/key
            duration = state.get('duration', 0.0) if isinstance(state, dict) else getattr(state, 'duration', 0.0)
            if duration is not None:
                cumulative_agent_times[player_idx] += duration

    # Extract final results
    final_step = env.steps[-1]
    match_results = []
    
    for player_idx, state in enumerate(final_step):
        # Extract just the filename for cleaner logging
        agent_name = match['agents'][player_idx].split('/')[-1]
        match_results.append({
            "player_id": player_idx,
            "agent": agent_name,
            "reward": state.reward,
            "status": state.status,
            "total_time": cumulative_agent_times[player_idx]
        })

    results_summary.append({
        "match": i,
        "mode": match['mode'],
        "seed": match['seed'],
        "wall_time": sim_duration,
        "details": match_results
    })

    # Render and save the specific HTML replay using the random seed in the filename
    html_output = env.render(mode="html", width=800, height=600)
    filename = f"./replays/replay_match_{i}_{match['mode']}_seed{match['seed']}.html"
    
    with open(filename, "w") as f:
        f.write(html_output)
        
    print(f"✅ Replay saved to {filename}\n")

# Print the final tournament summary
print("=" * 65)
print("🏆 FINAL TOURNAMENT RESULTS 🏆")
print("=" * 65)

for res in results_summary:
    print(f"\nMatch {res['match']} | {res['mode']} | Seed: {res['seed']} | Sim Time: {res['wall_time']:.2f}s")
    print("-" * 65)
    
    # Sort by reward descending to show the winner at the top
    sorted_players = sorted(res['details'], key=lambda x: (x['reward'] or -1), reverse=True)
    
    for player in sorted_players:
        reward_display = f"{player['reward']:.1f}" if player['reward'] is not None else "Elim."
        time_display = f"{player['total_time']:.3f}s"
        
        # Formatted output to clearly show the Time column
        print(f"Player {player['player_id']:<2} | {player['agent']:<25} | Score: {reward_display:>5} | CPU: {time_display:>8} ({player['status']})")
        
print("\nTournament execution finished. Check your directory for the HTML replays!")