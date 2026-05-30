# Orbit Wars Agent Development Tracker

## Overview
This document tracks the evolution, architectural shifts, and heuristic tuning of the `c-optimized-sniper` agent for the Kaggle Orbit Wars simulation competition. 

---

## Phase 1: The Baseline & The Speed Bottleneck
* **State:** Deployed the baseline `b-Snipe` heuristic agent.
* **The Problem:** The Python-based object-oriented geometry calculations were too heavy. The bot frequently hit the 1.0-second execution limit (`TIMEOUT`), causing it to freeze and skip critical defense and capture phases.
* **The Solution:** Completely refactored the physics engine. Stripped out heavy object instantiations in favor of tuples, arrays, and math-optimized distance functions (e.g., using squared distances to avoid `math.sqrt()` overhead).
* **Result:** Agent execution time dropped to near-zero milliseconds locally, ensuring 100% completion of the `plan_moves` pipeline.

## Phase 2: The "Bored General" Syndrome (Trickle Bleed)
* **State:** The optimized bot was inexplicably losing to the slow, timing-out baseline bot. 
* **The Investigation:** Built custom Exploratory Data Analysis (EDA) scripts in Jupyter to parse the embedded Kaggle JSON from the HTML replays. 
* **The Bug:** Because the optimized engine ran so fast, it always reached the `optional_phase` of the code. It used its spare time to calculate cross-map routes, trickling its entire planetary garrison away in tiny 8-ship fleets. The unoptimized bot survived purely because it timed out before it could make these terrible moves.
* **The Fix:** * Clamped `ATTACK_COST_TURN_WEIGHT` to `0.85` to heavily penalize long travel distances.
  * Raised `FOLLOWUP_MIN_SHIPS` to `22` to ensure the bot only launched actual threat-sized fleets, keeping the rest at home to compound interest.

## Phase 3: The Turtle vs. The Rusher (Format Parity)
* **State:** The anti-trickle fixes caused the bot to dominate 4-player Free-For-Alls (FFA) but lose 1v1 matches.
* **The Bug:** The bot became an uncrackable "Turtle." This is the optimal strategy in a 4-player game (survive while others fight), but in a 1v1, pure aggression and rapid expansion win. The bot was leaving 65% of its army sitting idle in the backlines during 1v1s.
* **The Fix:** "Split the brain" by making thresholds dynamic based on `world.is_four_player`.
  * **1v1:** Hyper-aggressive. Drops `PROACTIVE_DEFENSE_RATIO` to `~0.175`, raises supply chain flow to `0.62`, and implements a strict **25-Turn Distance Leash** on optional moves to ensure rapid, localized snowballing.
  * **FFA:** Paranoid turtle. Defense ratio at `0.35`, heavily restricted launch floors.

## Phase 4: The Phantom Threat (Physics Exploit)
* **State:** Observed the optimized bot launching bizarre 1-ship fleets at empty planets.
* **The Exploit:** Because Orbit Wars scales fleet speed logarithmically, a 1-ship fleet moves at the absolute minimum speed (1.0 unit/turn). 
* **The Impact:** These tiny fleets take ~60 turns to arrive. For those 60 turns, the unoptimized enemy bots register an "inbound threat" and waste their CPU cycles running deep 60-turn combat simulations. This effectively acts as a Denial of Service (DoS) attack on the enemy AI, causing them to time out and freeze while the optimized main army sweeps the board.

## Phase 5: Advanced Macro & Total War (Current State)
* **Comet Evacuation:** Hijacked the existing `live_doomed` retreat protocol. The bot now automatically detects when a comet is about to fly off the board and seamlessly evacuates all stationed ships to the nearest safe allied planet, preserving massive amounts of macro economy.
* **Total War:** Enabled `TOTAL_WAR` mode. At Turn 460, the bot abandons all defense and forcefully empties every planet at the enemy to maximize the exact ship-count win condition at Turn 500.
* **Hostile Reinforcements:** Enabled `HOSTILE_REINFORCE_ENABLED`. The bot now runs sub-simulations to predict if an enemy will send backup to a targeted planet, automatically packing extra ships to crush the anticipated defense.

---
**Next Implementation Targets:**
1. `FOUR_SOURCE_SWARM_ENABLED`: Activating 4-planet coordinated strikes for late-game siege breaking.
2. `WAIT_STRIKES`: Coding the intentional launch-delay logic to bypass enemy defense margins with synchronized multi-fleet impacts.