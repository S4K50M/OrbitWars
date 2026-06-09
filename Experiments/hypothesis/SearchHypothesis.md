# Experiment 1
# Experiments on Search: Orbit Wars Agent Evolution

This document tracks the iterative development of the search and decision-making logic for the Orbit Wars agent. The objective is to replace the baseline Greedy Selector with a more sophisticated, high-performance search algorithm that remains within the strict 1.0s inference budget.

## Experiment Log

| Experiment | Algorithm | Logic / Approach | Performance Result |
| :--- | :--- | :--- | :--- |
| **Baseline** | Greedy Selector | Direct selection of highest `net_ship_delta` per target. | Stable, but suffers in FFA/late game. |
| **Exp 1** | Hungarian (Bipartite) | Global assignment using `linear_sum_assignment`. | **Failure:** CPU/GPU sync bottleneck killed performance. |
| **Exp 2** | Sinkhorn-Knopp | Approximation of Hungarian, fully on-GPU. | **Mediocre:** Approximation noise led to erratic movement. |
| **Exp 3** | Phantom Fleets | Minimax-Lite: Predictive enemy injection into buckets. | **Success:** Beat baseline; handles local threats perfectly. |

---

## 1. The Bottleneck: The Hungarian/Sinkhorn Sync Tax
* **Hypothesis:** Replacing greedy selection with global optimal matching (Hungarian) would prevent sub-optimal ship allocation.
* **Failure Analysis:** The `scipy.optimize.linear_sum_assignment` (Hungarian) requires moving large tensor matrices from the GPU to the CPU, running the algorithm on the CPU, and pushing results back to the GPU. 
    * **Observation:** The "Memory Sync Tax" outweighed the efficiency gains of the algorithm.
    * **Sinkhorn-Knopp:** Even on-GPU, the approximation introduced "soft" assignment noise. In Orbit Wars, discrete ship integers matter. The soft-assignment approach was effectively "guessing" half-ship moves that the engine couldn't execute cleanly, causing tactical oscillation.

## 2. The Breakthrough: Phantom Fleet Prediction
* **Hypothesis:** Rather than attempting to find a global optimal match, we can maintain the Greedy algorithm but upgrade the *Information State* of the agent.
* **Implementation:** Before scoring our own moves, we simulate the enemy's most likely move. We inject these "Phantom Fleets" into the `fleet_buckets`.
* **Result:** This is essentially a **"Minimax-Lite"**. Because the greedy selector scores our moves against a board that *already includes the enemy's predicted attack*, the bot naturally chooses moves that are resilient to the enemy's plans.

## 3. Current Strategy: The Threat-Aware Gradient
* **Logic:** Instead of relying on a static reserve, we implemented `potential_attack_risk`.
* **Mechanism:** * The bot casts an analytic ray from every enemy planet to every friendly planet.
    * If an enemy path is clear (no sun-collision), that path is flagged as a "Risk Path."
    * The `safe_drain` function subtracts a `dynamic_risk_buffer` from the available ship pool based on this risk map.
* **Key Advantage:** This is a zero-sync, pure-tensor operation. It provides perfect, instantaneous defensive awareness without adding any latency to the `run_turn` loop.

---

## Lessons Learned & Future Hypothesis

### Lessons Learned
1.  **Greedy vs. Optimal:** In a continuous-physics game with 500 turns, **speed is the ultimate feature.** A greedy algorithm that runs in 10ms is vastly superior to a global optimizer that runs in 250ms, because the greedy version allows for deeper forward projections (higher `horizon`) and more candidates.
2.  **Stochastic Robustness:** The move toward "Phantom Fleets" teaches us that predicting *enemy intent* is more valuable than perfectly matching *our own ships*.

### Future Research: "Stochastic Ensemble Phantoms"
* **Hypothesis:** Current Phantom logic predicts the enemy's #1 move. If the enemy has three high-value choices, we over-fit our defense to the #1 choice. 
* **Proposed Experiment:** Inject a weighted "Probability Cloud" of the Top 3 enemy moves into the `fleet_buckets` instead of just the Top 1. 
* **Expected Outcome:** This will create a probabilistic garrison projection, making the bot significantly harder to "bait" into committing to the wrong defensive sector.