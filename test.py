import math
import time
import random
import os
from collections import defaultdict
from kaggle_environments import make

# Ensure replays directory exists
os.makedirs("./replays", exist_ok=True)

# Define our primary agent
MAIN_AGENT_NAME = "f-aurora-forge"
main_agent = f"./agents/{MAIN_AGENT_NAME}.py"

# Define the gauntlet of opponents
opponents = [
    "./agents/e-tankman.py",
    "./agents/d-eclipse-predator.py",
    "./Testing-agents/apex-predator.py",
    "./agents/c-optimized-sniper.py",
    "./agents/b-Snipe+No-BOM.py"
]

def generate_tournament_matches():
    matches = []
    match_id = 1
    
    # Bracket 1: 1v1 Duels (1 match against each opponent)
    for opp in opponents:
        seed = random.randint(100000, 999999)
        agents = [{"path": main_agent, "is_main": True}, {"path": opp, "is_main": False}]
        random.shuffle(agents) # Prevent Player 0 bias
        matches.append({"id": match_id, "mode": "2-player", "type": "1v1 Duel", "seed": seed, "agents": agents})
        match_id += 1

    # Bracket 2: Clone War FFA (Main vs 3 copies of the same opponent)
    for opp in opponents:
        seed = random.randint(100000, 999999)
        agents = [{"path": main_agent, "is_main": True}] + [{"path": opp, "is_main": False} for _ in range(3)]
        random.shuffle(agents)
        matches.append({"id": match_id, "mode": "4-player", "type": "Clone FFA", "seed": seed, "agents": agents})
        match_id += 1
        
    # Bracket 3: Chaos FFA (Main vs 3 different random opponents)
    for _ in range(5):
        seed = random.randint(100000, 999999)
        mixed_opps = random.sample(opponents, 3)
        agents = [{"path": main_agent, "is_main": True}] + [{"path": p, "is_main": False} for p in mixed_opps]
        random.shuffle(agents)
        matches.append({"id": match_id, "mode": "4-player", "type": "Chaos FFA", "seed": seed, "agents": agents})
        match_id += 1

    return matches

matches = generate_tournament_matches()
results_summary = []
main_agent_stats = {"wins": 0, "losses": 0, "1v1_wins": 0, "ffa_wins": 0}

print("=" * 70)
print(f"🚀 STARTING AURORA FORGE GAUNTLET ({len(matches)} MATCHES) 🚀")
print("=" * 70 + "\n")

for match in matches:
    print(f"Running Match {match['id']} | {match['type']} | Seed: {match['seed']}...")
    
    agent_paths = [a["path"] for a in match["agents"]]
    env = make("orbit_wars", configuration={"seed": match['seed']}, debug=True)

    start_time = time.time()
    env.run(agent_paths)
    sim_duration = time.time() - start_time
    
    print(f"  -> Simulation completed in {sim_duration:.2f} seconds.")

    # Calculate cumulative processing time
    num_players = len(agent_paths)
    cumulative_agent_times = [0.0] * num_players
    
    for step in env.steps[1:]:
        for player_idx, state in enumerate(step):
            duration = state.get('duration', 0.0) if isinstance(state, dict) else getattr(state, 'duration', 0.0)
            if duration is not None:
                cumulative_agent_times[player_idx] += duration

    # Extract final results
    final_step = env.steps[-1]
    match_results = []
    
    # Determine the winning score (accounting for floats)
    max_score = max((state.reward for state in final_step if state.reward is not None), default=-999)
    
    for player_idx, state in enumerate(final_step):
        agent_name = agent_paths[player_idx].split('/')[-1]
        is_main = match["agents"][player_idx]["is_main"]
        is_winner = state.reward == max_score and state.reward is not None
        
        if is_main:
            if is_winner:
                main_agent_stats["wins"] += 1
                if match["mode"] == "2-player":
                    main_agent_stats["1v1_wins"] += 1
                else:
                    main_agent_stats["ffa_wins"] += 1
            else:
                main_agent_stats["losses"] += 1

        match_results.append({
            "player_id": player_idx,
            "agent": agent_name,
            "is_main": is_main,
            "reward": state.reward,
            "status": state.status,
            "total_time": cumulative_agent_times[player_idx],
            "is_winner": is_winner
        })

    results_summary.append({
        "match": match['id'],
        "type": match['type'],
        "seed": match['seed'],
        "wall_time": sim_duration,
        "details": match_results
    })

    # Render and save
    html_output = env.render(mode="html", width=800, height=600)
    filename = f"./replays/match_{match['id']}_{match['type'].replace(' ', '')}_seed{match['seed']}.html"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_output)
        
    print(f"✅ Replay saved to {filename}\n")

# ==========================================
# FINAL TOURNAMENT REPORT
# ==========================================
print("\n" + "=" * 75)
print("🏆 TOURNAMENT MATCH LOG 🏆")
print("=" * 75)

for res in results_summary:
    print(f"\nMatch {res['match']} | {res['type']} | Seed: {res['seed']} | Sim Time: {res['wall_time']:.2f}s")
    print("-" * 75)
    
    sorted_players = sorted(res['details'], key=lambda x: (x['reward'] or -999), reverse=True)
    
    for p in sorted_players:
        reward_display = f"{p['reward']:.1f}" if p['reward'] is not None else "Elim."
        time_display = f"{p['total_time']:.3f}s"
        tag = "👑 WINNER" if p['is_winner'] else ""
        main_flag = ">> " if p['is_main'] else "   "
        
        print(f"{main_flag}P{p['player_id']} | {p['agent']:<25} | Score: {reward_display:>5} | CPU: {time_display:>8} ({p['status']}) {tag}")

print("\n" + "=" * 75)
print(f"📊 {MAIN_AGENT_NAME} PERFORMANCE ANALYTICS 📊")
print("=" * 75)
total_matches = len(matches)
win_rate = (main_agent_stats["wins"] / total_matches) * 100
win_1v1 = (main_agent_stats["1v1_wins"] / 5) * 100
win_ffa = (main_agent_stats["ffa_wins"] / 10) * 100

print(f"Overall Win Rate:  {win_rate:.1f}% ({main_agent_stats['wins']}/{total_matches})")
print(f"1v1 Duel Win Rate: {win_1v1:.1f}% ({main_agent_stats['1v1_wins']}/5)")
print(f"FFA Win Rate:      {win_ffa:.1f}% ({main_agent_stats['ffa_wins']}/10)")
print("=" * 75)