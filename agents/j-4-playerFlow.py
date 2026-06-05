import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field, replace

# ============================================================
# Shared Configuration (incorporating all meta updates)
# ============================================================

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
TOTAL_STEPS = 500
SIM_HORIZON = 110
ROUTE_SEARCH_HORIZON = 60
HORIZON = 180
LAUNCH_CLEARANCE = 0.1
FLEET_SWEEP_HORIZON = 90

EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 50                     # Tick‑50 transition
LATE_REMAINING_TURNS = 60
VERY_LATE_REMAINING_TURNS = 25
TOTAL_WAR_ENABLED = False
TOTAL_WAR_REMAINING_TURNS = 40
TOTAL_WAR_MIN_SEND = 5

SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
INTERCEPT_TOLERANCE = 1

SAFE_OPENING_PROD_THRESHOLD = 4
SAFE_OPENING_TURN_LIMIT = 10
ROTATING_OPENING_MAX_TURNS = 13
ROTATING_OPENING_LOW_PROD = 2
FOUR_PLAYER_ROTATING_REACTION_GAP = 3
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.72
FOUR_PLAYER_ROTATING_TURN_LIMIT = 14

COMET_MAX_CHASE_TURNS = 15

ATTACK_COST_TURN_WEIGHT = 0.85
SNIPE_COST_TURN_WEIGHT = 0.45
INDIRECT_VALUE_SCALE = 0.15
INDIRECT_FRIENDLY_WEIGHT = 0.35
INDIRECT_NEUTRAL_WEIGHT = 0.9
INDIRECT_ENEMY_WEIGHT = 1.25

STATIC_NEUTRAL_VALUE_MULT = 1.4
STATIC_HOSTILE_VALUE_MULT = 1.6
ROTATING_OPENING_VALUE_MULT = 0.9
HOSTILE_TARGET_VALUE_MULT = 2.05
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.48
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
EARLY_NEUTRAL_VALUE_MULT = 1.2
COMET_VALUE_MULT = 0.65
SNIPE_VALUE_MULT = 1.20
SWARM_VALUE_MULT = 1.05
REINFORCE_VALUE_MULT = 1.35
CRASH_EXPLOIT_VALUE_MULT = 1.18
FINISHING_HOSTILE_VALUE_MULT = 1.15
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92

# Apex Predator new constants
FFA_OPPORTUNISM_ENABLED = True
EXPOSED_OUTBOUND_RATIO = 0.7
EXPOSED_PLANET_VALUE_MULT = 3.0
EXPOSED_PLANET_SCORE_MULT = 3.0
BLOOD_IN_WATER_VALUE_MULT = 1.28
WEAKEST_ENEMY_VALUE_MULT = 1.12
LET_THEM_FIGHT_PENALTY = 0.6
LEADER_DENIAL_ENABLED = False
LEADER_DENIAL_PROD_GAP = 4
LEADER_DENIAL_STRENGTH_GAP = 30
LEADER_DENIAL_VALUE_MULT = 1.35
LEADER_DENIAL_SCORE_MULT = 1.12
LEADER_DENIAL_PRODUCTION_BONUS = 1.8

TRAP_VULNERABLE_THRESHOLD = 5
TRAP_TRAVEL_LIMIT = 10
TRAP_GARRISON_MULT = 2.5
TRAP_PENALTY = 0.9

DYNAMIC_DEFENSE_HIGH_RATIO = 0.85
DYNAMIC_DEFENSE_LOW_RATIO = 0.10
DYNAMIC_DEFENSE_THREAT_TURNS = 10
DYNAMIC_DEFENSE_THREAT_SHIPS = 50
DYNAMIC_DEFENSE_SAFE_TURNS = 25

ENDGAME_TURN_THRESHOLD = 30
ENDGAME_POINT_PADDING_TURNS = 60

MACRO_SHIP_COST_WEIGHT = 1.5

NEUTRAL_MARGIN_BASE = 2
NEUTRAL_MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 12
HOSTILE_MARGIN_BASE = 3
HOSTILE_MARGIN_PROD_WEIGHT = 2
HOSTILE_MARGIN_CAP = 18
HOSTILE_REINFORCE_ENABLED = False
HOSTILE_REINFORCE_HORIZON = 8
HOSTILE_REINFORCE_RATIO = 0.22
HOSTILE_REINFORCE_CAP = 12
STATIC_TARGET_MARGIN = 4
CONTESTED_TARGET_MARGIN = 5
FOUR_PLAYER_TARGET_MARGIN = 5
LONG_TRAVEL_MARGIN_START = 18
LONG_TRAVEL_MARGIN_DIVISOR = 3
LONG_TRAVEL_MARGIN_CAP = 8
COMET_MARGIN_RELIEF = 6
FINISHING_HOSTILE_SEND_BONUS = 3

STATIC_TARGET_SCORE_MULT = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.92
DENSE_STATIC_NEUTRAL_COUNT = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.86
SNIPE_SCORE_MULT = 1.18
SWARM_SCORE_MULT = 1.06
CRASH_EXPLOIT_SCORE_MULT = 1.05
BLOOD_IN_WATER_SCORE_MULT = 1.12
WEAKEST_ENEMY_SCORE_MULT = 1.08

FOLLOWUP_MIN_SHIPS = 22
LOW_VALUE_COMET_PRODUCTION = 1
LATE_CAPTURE_BUFFER = 5
VERY_LATE_CAPTURE_BUFFER = 3

DEFENSE_LOOKAHEAD_TURNS = 28
DEFENSE_COST_TURN_WEIGHT = 0.4
DEFENSE_FRONTIER_SCORE_MULT = 1.12
DEFENSE_SEND_MARGIN_BASE = 1
DEFENSE_SEND_MARGIN_PROD_WEIGHT = 1
DEFENSE_SHIP_VALUE = 0.55

REINFORCE_ENABLED = True
REINFORCE_MIN_PRODUCTION = 2
REINFORCE_MAX_TRAVEL_TURNS = 22
REINFORCE_SAFETY_MARGIN = 2
REINFORCE_MAX_SOURCE_FRACTION = 0.75
REINFORCE_MIN_FUTURE_TURNS = 40
REINFORCE_HOLD_LOOKAHEAD = 20
REINFORCE_COST_TURN_WEIGHT = 0.35

RECAPTURE_LOOKAHEAD_TURNS = 10
RECAPTURE_COST_TURN_WEIGHT = 0.52
RECAPTURE_VALUE_MULT = 0.88
RECAPTURE_FRONTIER_MULT = 1.08
RECAPTURE_PRODUCTION_WEIGHT = 0.6
RECAPTURE_IMMEDIATE_WEIGHT = 0.4

REAR_SOURCE_MIN_SHIPS = 16
REAR_DISTANCE_RATIO = 1.25
REAR_STAGE_PROGRESS = 0.78
REAR_SEND_RATIO_TWO_PLAYER = 0.8
REAR_SEND_RATIO_FOUR_PLAYER = 0.6
REAR_SEND_MIN_SHIPS = 10
REAR_MAX_TRAVEL_TURNS = 40

PARTIAL_SOURCE_MIN_SHIPS = 6
MULTI_SOURCE_TOP_K = 10
MULTI_SOURCE_ETA_TOLERANCE = 2
MULTI_SOURCE_PLAN_PENALTY = 0.97
HOSTILE_SWARM_ETA_TOLERANCE = 1
THREE_SOURCE_SWARM_ENABLED = True
THREE_SOURCE_MIN_TARGET_SHIPS = 20
THREE_SOURCE_ETA_TOLERANCE = 2
THREE_SOURCE_PLAN_PENALTY = 0.94

WAIT_STRIKE_ENABLED = True
WAIT_STRIKE_DELAYS = (0, 2, 4, 6)
WAIT_STRIKE_MAX_TARGETS = 6

FOUR_SOURCE_SWARM_ENABLED = True
FOUR_SOURCE_ETA_TOLERANCE = 2
FOUR_SOURCE_MIN_TARGET_SHIPS = 40
FOUR_SOURCE_PLAN_PENALTY = 0.91

PROACTIVE_DEFENSE_HORIZON = 12
PROACTIVE_DEFENSE_RATIO = 0.4
MULTI_ENEMY_PROACTIVE_HORIZON = 14
MULTI_ENEMY_PROACTIVE_RATIO = 0.22
MULTI_ENEMY_STACK_WINDOW = 3
REACTION_SOURCE_TOP_K_MY = 4
REACTION_SOURCE_TOP_K_ENEMY = 4
PROACTIVE_ENEMY_TOP_K = 3

CRASH_EXPLOIT_ENABLED = True
CRASH_EXPLOIT_MIN_TOTAL_SHIPS = 10
CRASH_EXPLOIT_ETA_WINDOW = 2
CRASH_EXPLOIT_POST_CRASH_DELAY = 1

LATE_IMMEDIATE_SHIP_VALUE = 0.6
WEAK_ENEMY_THRESHOLD = 45
ELIMINATION_BONUS = 18.0
FFA_ELIMINATION_SHIPS = 55
FFA_LET_FIGHT_MIN_SHIPS = 14
EXPOSED_OUTBOUND_MIN_SHIPS = 12

LONG_TRAVEL_PENALTY_TURNS = 35
LONG_TRAVEL_PENALTY_MULT = 0.5  # Harsh enough to prioritize local, soft enough to allow cross-map

BEHIND_DOMINATION = -0.20
AHEAD_DOMINATION = 0.18
FINISHING_DOMINATION = 0.35
FINISHING_PROD_RATIO = 1.25
AHEAD_ATTACK_MARGIN_BONUS = 0.08
BEHIND_ATTACK_MARGIN_PENALTY = 0.05
FINISHING_ATTACK_MARGIN_BONUS = 0.08

DOOMED_EVAC_TURN_LIMIT = 24
DOOMED_MIN_SHIPS = 8

SOFT_ACT_DEADLINE = 0.82
HEAVY_PHASE_MIN_TIME = 0.16
OPTIONAL_PHASE_MIN_TIME = 0.08
HEAVY_ROUTE_PLANET_LIMIT = 32

# ============================================================
# Potential Field / Gravity Configuration
# ============================================================
GRAVITY_GAMMA = 2.2         # Balances local clustering with opportunistic snipes
COST_PENALTY_EXP = 0.9      # Penalize expensive targets
MACRO_SHIP_COST_EARLY = 1.2 # Cost weight for first 150 turns
MACRO_SHIP_COST_LATE = 0.6  # More expendable late game
LATE_GAME_STEP = 150        

# Strategic Gravity Multipliers
M_RESCUE = 3.5              # MASSIVE priority to save our own falling planets
M_NEUTRAL = 1.8             # High priority for free map control
M_WEAKEST_ENEMY = 1.35      
M_LEADER_DENIAL = 1.25      
M_STANDARD_ENEMY = 1.0

# FFA Modifiers
FFA_HOSTILE_COST_MULT = 2.5 # In FFA, fighting enemies is mathematically punished unless necessary
M_HEAT_VULTURE = 3.0        # Massive pull if enemies are bleeding each other

# Defensive & Logistical Parameters
FRONT_LINE_DIST = 45.0      
MICRO_BUFFER_FFA = 15       # 🚨 Stronger base shield in FFA to stop 1-ship steals
MICRO_BUFFER_1V1 = 5
EARLY_GAME_HORIZON = 25     
VANGUARD_TOLERANCE = 12     # 🚨 Swarms cannot span more than 12 turns difference in ETA


_log1000 = math.log(1000)
SPEED_CACHE = []
for i in range(1,1001):
    SPEED_CACHE.append(1+(MAX_SPEED-1)*(math.log(i)/_log1000)**1.5)

# ============================================================
# Shared Types
# ============================================================

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

@dataclass(frozen=True, slots=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None

@dataclass(slots=True)
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[ShotOption] = field(default_factory=list)

# ============================================================
# Physics (Dynamic INTERCEPT_TOLERANCE)
# ============================================================

def dist(ax, ay, bx, by):
    return math.sqrt((ax - bx)*(ax-bx)+(ay - by)*(ay-by))

def dist_sq(ax,ay,bx,by):
    return (ax-bx)*(ax-bx)+(ay-by)*(ay-by)

def orbital_radius(planet):
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)

def orbital_radius_sq(planet):
    return dist_sq(planet.x, planet.y, CENTER_X, CENTER_Y)

def is_static_planet(planet):
    return orbital_radius_sq(planet) >= (ROTATION_LIMIT-planet.radius)**2

def fleet_speed(ships):
    ships = int(ships)
    if ships <= 1:
        return 1.0
    elif ships>= 1000:
        return MAX_SPEED
    return SPEED_CACHE[ships-1]

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)

def point_to_segment_distance_sq(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist_sq(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist_sq(px, py, proj_x, proj_y)

def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_distance_sq(CENTER_X, CENTER_Y, x1, y1, x2, y2) < (SUN_R + safety)**2

def launch_point(sx, sy, sr, angle):
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance

def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance

def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx, sy, sr, tx, ty, tr,
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance

def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r2 = dist_sq(init.x, init.y, CENTER_X, CENTER_Y)
    if r2 >= (ROTATION_LIMIT-init.radius)**2:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    r = math.sqrt(r2)
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )

def predict_comet_position(planet_id, comets, turns):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None

def comet_remaining_life(planet_id, comets):
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0

def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns

def travel_time(sx, sy, sr, tx, ty, tr, ships):
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        return 1e9
    return est[1]

def predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids):
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)

def target_can_move(target, initial_by_id, comet_ids):
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    r = dist_sq(init.x, init.y, CENTER_X, CENTER_Y)
    return r<(ROTATION_LIMIT-init.radius)**2

def search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    best = None
    best_score = None
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))

    tol = 0.1 if target_can_move(target, initial_by_id, comet_ids) else INTERCEPT_TOLERANCE

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target, candidate_turns, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            continue
        est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > tol:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_target_position(
            target, actual_turns, initial_by_id, ang_vel, comets, comet_ids)
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x, src.y, src.radius, actual_pos[0], actual_pos[1], target.radius, ships)
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > tol:
            continue

        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best

def aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        if not target_can_move(target, initial_by_id, comet_ids):
            return None
        return search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)

    tx, ty = target.x, target.y
    can_move = target_can_move(target, initial_by_id, comet_ids)
    tol = 0.1 if can_move else INTERCEPT_TOLERANCE

    for _ in range(5):
        _, turns = est
        pos = predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            if not can_move:
                return None
            return search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)
        if (abs(ntx - tx) < 0.3 and abs(nty - ty) < 0.3 and abs(next_est[1] - turns) <= tol):
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)
    return final_est[0], final_est[1], tx, ty

def fleet_target_planet(fleet, planets, initial_by_id=None, ang_vel=0.0, comets=(), comet_ids=()):
    initial_by_id = initial_by_id or {}
    comet_ids = set(comet_ids or ())
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)
    moving_targets = []

    for planet in planets:
        if planet.id == fleet.from_planet_id:
            continue
        if target_can_move(planet, initial_by_id, comet_ids):
            moving_targets.append(planet)
            continue

        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if moving_targets:
        max_turns = min(HORIZON, FLEET_SWEEP_HORIZON)
        prev_x = fleet.x
        prev_y = fleet.y
        viable_moving_targets = []
        for planet in moving_targets:
            dx = planet.x - fleet.x
            dy = planet.y - fleet.y
            max_dist = (speed * max_turns) + planet.radius + 15
            if dx*dx + dy*dy > max_dist * max_dist:
                continue
            if dx * dir_x + dy * dir_y < -planet.radius - 15:
                continue
            viable_moving_targets.append(planet)
        for turn in range(1, max_turns + 1):
            if turn > best_time:
                break
            cur_x = fleet.x + dir_x * speed * turn
            cur_y = fleet.y + dir_y * speed * turn
            for planet in viable_moving_targets:
                pos = predict_target_position(planet, turn, initial_by_id, ang_vel, comets, comet_ids)
                if pos is None:
                    continue
                if (point_to_segment_distance(pos[0], pos[1], prev_x, prev_y, cur_x, cur_y) <= planet.radius
                        and turn < best_time):
                    best_time = turn
                    best_planet = planet
            prev_x = cur_x
            prev_y = cur_y

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))

# --- Corrected build_arrival_ledger with noise filter ---
def build_arrival_ledger(fleets, planets, initial_by_id=None, ang_vel=0.0, comets=(), comet_ids=()):
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(
            fleet, planets, initial_by_id=initial_by_id,
            ang_vel=ang_vel, comets=comets, comet_ids=comet_ids)
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))

    # Noise filter: Prune micro-fleets (<5 ships) if they are spam (>15 fleets) and low mass (<30% garrison)
    for pid in list(arrivals_by_planet.keys()):
        arrivals = arrivals_by_planet[pid]
        planet = next((p for p in planets if p.id == pid), None)
        if planet is None:
            continue
        
        new_arrivals = []
        owner_events = defaultdict(list)
        
        # Group by owner, keeping the full tuple
        for event in arrivals:
            eta, owner, ships = event
            owner_events[owner].append(event)
            
        for owner, events in owner_events.items():
            small_fleets = [e for e in events if e[2] < 5]
            
            # If enemy is micro-spamming this planet
            if owner != planet.owner and len(small_fleets) > 15:
                total_small_mass = sum(e[2] for e in small_fleets)
                if total_small_mass < planet.ships * 0.3:
                    # Drop the small fleets, keep only the large ones (ships >= 5)
                    new_arrivals.extend([e for e in events if e[2] >= 5])
                    continue
            
            # Otherwise, keep all fleets from this owner
            new_arrivals.extend(events)
            
        arrivals_by_planet[pid] = new_arrivals

    return arrivals_by_planet

def resolve_arrival_event(owner, garrison, arrivals):
    by_owner = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison

def normalize_arrivals(arrivals, horizon):
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events

def simulate_planet_timeline(planet, arrivals, player, horizon):
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn
        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True
    if planet.owner == player:
        def survives_with_keep(keep):
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(sim_owner, sim_garrison, group)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }

def state_at_timeline(timeline, arrival_turn):
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)

def count_players(planets, fleets):
    owners = set()
    for planet in planets:
        if planet.owner != -1:
            owners.add(planet.owner)
    for fleet in fleets:
        owners.add(fleet.owner)
    return max(2, len(owners))

def nearest_distance_to_set(px, py, planets):
    if not planets:
        return 10**9
    return min(dist_sq(px, py, planet.x, planet.y) for planet in planets)

def indirect_features(planet, planets, player):
    friendly = 0.0
    neutral = 0.0
    enemy = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 5.0)
        if other.owner == player:
            friendly += factor
        elif other.owner == -1:
            neutral += factor
        else:
            enemy += factor
    return friendly, neutral, enemy

def detect_exposed_enemy_planets(fleets, enemy_planets):
    exposed = set()
    for planet in enemy_planets:
        outbound = sum(
            int(fleet.ships)
            for fleet in fleets
            if fleet.owner == planet.owner
            and fleet.from_planet_id == planet.id
            and fleet.ships >= 5
        )
        total_visible = planet.ships + outbound
        if total_visible > 0 and outbound >= total_visible * EXPOSED_OUTBOUND_RATIO:
            exposed.add(planet.id)
    return exposed

def detect_enemy_fights(arrivals_by_planet, player):
    contested = {}
    for planet_id, arrivals in arrivals_by_planet.items():
        enemy_owners = set()
        enemy_ships = 0
        for _, owner, ships in arrivals:
            if owner not in (-1, player):
                enemy_owners.add(owner)
                enemy_ships += int(ships)
        if len(enemy_owners) >= 2 and enemy_ships >= FFA_LET_FIGHT_MIN_SHIPS:
            contested[planet_id] = enemy_ships
    return contested


def planet_distance_sq(first, second):
    dx = first.x - second.x
    dy = first.y - second.y
    return dx*dx+dy*dy


class WorldModel:
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id = {planet.id: planet for planet in planets}
        self.my_planets = [planet for planet in planets if planet.owner == player]
        self.enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        self.neutral_planets = [planet for planet in planets if planet.owner == -1]
        self.static_neutral_planets = [
            planet for planet in self.neutral_planets if is_static_planet(planet)
        ]

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for planet in planets:
            if planet.owner != -1:
                self.owner_strength[planet.owner] += int(planet.ships)
                self.owner_production[planet.owner] += int(planet.production)
        for fleet in fleets:
            self.owner_strength[fleet.owner] += int(fleet.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_total = sum(
            strength for owner, strength in self.owner_strength.items() if owner != player
        )
        
        enemy_owners = [owner for owner in self.owner_strength if owner != player]
        if enemy_owners:
            self.weakest_enemy = min(enemy_owners, key=lambda owner: self.owner_strength[owner])
            self.leader_enemy = max(
                enemy_owners,
                key=lambda owner: (self.owner_production.get(owner, 0), self.owner_strength[owner]),
            )
        else:
            self.weakest_enemy = None
            self.leader_enemy = None

        self.blood_in_water_owners = {
            owner for owner in enemy_owners
            if self.owner_strength[owner] <= FFA_ELIMINATION_SHIPS
        }
        
        self.arrivals_by_planet = build_arrival_ledger(
            fleets, planets, initial_by_id=initial_by_id, ang_vel=ang_vel,
            comets=comets, comet_ids=comet_ids)
            
        self.base_timeline = {
            planet.id: simulate_planet_timeline(
                planet, self.arrivals_by_planet[planet.id], player, HORIZON)
            for planet in planets
        }
        self.keep_needed_map = {pid: self.base_timeline[pid]["keep_needed"] for pid in self.base_timeline}
        
        if FFA_OPPORTUNISM_ENABLED:
            self.enemy_fights = detect_enemy_fights(self.arrivals_by_planet, player)
        else:
            self.enemy_fights = {}

        # ============================================================
        # INFLUENCE MAP: Used strictly for internal Supply Line routing
        # ============================================================
        self.influence_map = {}
        if self.enemy_planets:
            for p in self.my_planets:
                min_dist = min(dist(p.x, p.y, e.x, e.y) for e in self.enemy_planets)
                self.influence_map[p.id] = min_dist
        else:
            for p in self.my_planets:
                self.influence_map[p.id] = 1e9

        self.shot_cache = {}
        self.exact_need_cache = {}
        self.cost_weight = MACRO_SHIP_COST_LATE if self.step > LATE_GAME_STEP else MACRO_SHIP_COST_EARLY

    def is_static(self, planet_id):
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        return comet_remaining_life(planet_id, self.comets)

    def plan_shot(self, src_id, target_id, ships):
        ships = int(ships)
        key = (src_id, target_id, ships)
        if (cached := self.shot_cache.get(key)) is not None:
            return cached
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        result = aim_with_prediction(
            src, target, ships, self.initial_by_id, self.ang_vel, self.comets, self.comet_ids)
        self.shot_cache[key] = result
        return result

    def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)
        arrivals = [item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= cutoff]
        arrivals.extend(item for item in planned_commitments.get(target_id, []) if item[0] <= cutoff)
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)
        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return state_at_timeline(dyn, cutoff)

    def min_ships_to_own_by(self, target_id, eval_turn, attacker_owner, arrival_turn=None, planned_commitments=None):
        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        
        owner_before, ships_before = self.projected_state(
            target_id, eval_turn, planned_commitments=planned_commitments)
            
        if owner_before == attacker_owner:
            return 0

        def owns_at(ships):
            owner_after, _ = self.projected_state(
                target_id, eval_turn, planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, attacker_owner, int(ships)),))
            return owner_after == attacker_owner

        hi = max(1, int(math.ceil(ships_before)) + 1)
        search_cap = 1000
        while hi <= search_cap and not owns_at(hi):
            hi *= 2
        if hi > search_cap:
            return hi
            
        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid): hi = mid
            else: lo = mid + 1
        return lo
    
# ============================================================
# Target-Centric Gravity Engine & Flow Calculation
# ============================================================

# ============================================================
# Comet Trojan & Trajectory Module
# ============================================================
def get_trojan_comets(world):
    """
    Identifies owned comets that are on an intercept trajectory with enemy territory.
    Returns a dict mapping comet_id -> Trojan Data.
    """
    trojans = {}
    for group in world.comets:
        path_index = group.get("path_index", 0)
        paths = group.get("paths", [])
        
        for idx, cid in enumerate(group.get("planet_ids", [])):
            planet = world.planet_by_id.get(cid)
            if not planet or planet.owner != world.player: continue
            if idx >= len(paths): continue
            
            path = paths[idx]
            life_left = len(path) - path_index
            if life_left <= 0: continue
            
            closest_enemy_dist = 1e9
            time_to_strike = -1
            target_enemy = None
            
            for t in range(min(35, life_left)):
                fx, fy = path[path_index + t]
                for enemy in world.enemy_planets:
                    if enemy.ships < 15 and enemy.production < 3: continue
                    
                    d = dist(fx, fy, enemy.x, enemy.y)
                    if d < closest_enemy_dist:
                        closest_enemy_dist = d
                        time_to_strike = t
                        target_enemy = enemy
                        
            if closest_enemy_dist <= 22.0 and target_enemy:
                trojans[cid] = {
                    "strike_t": time_to_strike,
                    "target": target_enemy,
                    "dist": closest_enemy_dist,
                    "life_left": life_left
                }
    return trojans

def build_defense_policy(world):
    reserve, attack_budget = {}, {}
    buffer_size = MICRO_BUFFER_FFA if world.is_four_player else MICRO_BUFFER_1V1
    
    for p in world.my_planets:
        base_reserve = world.keep_needed_map.get(p.id, 0)
        is_frontline = world.influence_map.get(p.id, 1e9) <= FRONT_LINE_DIST
        reserve[p.id] = min(int(p.ships), int(max(base_reserve, buffer_size) if is_frontline else base_reserve))
        attack_budget[p.id] = max(0, int(p.ships) - reserve[p.id])
    return reserve, attack_budget

def calculate_target_gravity(target, world, planned_commitments, attack_budget, spent_total):
    active_turns = max(0, world.remaining_steps - 10) 
    if active_turns <= 0: return None
    
    # 🚨 RESCUE INTEGRATION: Save our own falling planets!
    is_rescue = False
    if target.owner == world.player:
        if world.base_timeline[target.id]["holds_full"]: return None # It's safe
        is_rescue = True
    
    # Value Calculation
    if is_rescue:
        prod_value = target.production * 3.0 # Highly value saving our own economy
        m_strat = M_RESCUE
    else:
        prod_value = target.production * (2.0 if target.owner not in (-1, world.player) else 1.0)
        if target.id in world.comet_ids: prod_value = 1.5 
            
        m_strat = M_STANDARD_ENEMY
        if target.owner == -1: m_strat = M_NEUTRAL
        elif target.owner == world.weakest_enemy: m_strat = M_WEAKEST_ENEMY
        elif target.owner == world.leader_enemy: m_strat = M_LEADER_DENIAL
        
        # 🚨 THE COLD WAR MULTIPLIER: Punish fighting enemies in FFA
        if world.is_four_player and target.owner not in (-1, world.player):
            prod_value /= FFA_HOSTILE_COST_MULT
            
        if target.id in world.enemy_fights: m_strat *= M_HEAT_VULTURE
    
    raw_value = prod_value * m_strat * 1000.0

    # Gather & Sort Valid Sources
    available_sources = []
    for src in world.my_planets:
        if src.id == target.id: continue
        avail = attack_budget.get(src.id, 0) - spent_total[src.id]
        if avail < 5: continue 
        
        aim = world.plan_shot(src.id, target.id, avail)
        if aim:
            angle, turns, _, _ = aim
            if world.is_early and turns > EARLY_GAME_HORIZON: continue
            if turns > world.remaining_steps - 2: continue
            # 🚨 VOID CHECK: Ensure comets don't die before we land
            if target.id in world.comet_ids and world.comet_life(target.id) <= turns + 3: continue
            
            available_sources.append({'src': src, 'avail': avail, 'turns': turns, 'angle': angle})
            
    if not available_sources: return None
    available_sources.sort(key=lambda x: x['turns'])
    
    # 🚨 VANGUARD CLUSTERING: Enforce strict ETA grouping
    accumulated_ships = 0
    participating = []
    max_turns = 0
    needed = 0
    base_eta = available_sources[0]['turns']
    
    for src_data in available_sources:
        # Prevent distant planets from stalling the fast vanguard fleets
        if src_data['turns'] - base_eta > VANGUARD_TOLERANCE: continue
        
        participating.append(src_data)
        accumulated_ships += src_data['avail']
        max_turns = max(max_turns, src_data['turns'])
        
        # Vulture Delay Logic
        sync_turn = max_turns + (3 if target.id in world.enemy_fights and world.is_four_player else 0)
        needed = world.min_ships_to_own_by(target.id, sync_turn, world.player, arrival_turn=sync_turn, planned_commitments=planned_commitments)
        
        # Anti-Trap Padding
        if target.owner == -1 and not is_rescue:
            max_threat = 0
            for e in world.enemy_planets:
                if travel_time(e.x, e.y, e.radius, target.x, target.y, target.radius, 1) <= sync_turn + 5:
                    threat = int(e.production * (2 if world.is_four_player else 4))
                    if threat > max_threat: max_threat = threat
            needed += max_threat
        
        if accumulated_ships >= needed: break
            
    if accumulated_ships < needed or needed <= 0: return None 
        
    gravity = raw_value / ( ((needed + 1) ** COST_PENALTY_EXP) * (max_turns ** GRAVITY_GAMMA) )
    
    return {"gravity": gravity, "target_id": target.id, "max_turns": max_turns, "needed": needed, "participants": participating}

def plan_moves(world, deadline=None):
    def expired(): return deadline is not None and time.perf_counter() > deadline

    reserve, attack_budget = build_defense_policy(world)
    planned_commitments = defaultdict(list)
    spent_total = defaultdict(int)
    moves = []

    def append_move(src_id, angle, ships):
        moves.append([src_id, float(angle), int(ships)])
        return ships

    def source_inventory_left(source_id):
        return max(0, attack_budget.get(source_id, 0) - spent_total[source_id])

    trojans = get_trojan_comets(world)

    # --------------------------------------------------------
    # STAGE 0: The Executioner
    # --------------------------------------------------------
    if not world.is_four_player:
        planet_counts = defaultdict(int)
        for p in world.planets:
            if p.owner not in (-1, world.player): planet_counts[p.owner] += 1
                
        for owner, count in planet_counts.items():
            if count == 1:
                last_planet = next(p for p in world.planets if p.owner == owner)
                best_src, best_dist = None, 1e9
                for src in world.my_planets:
                    avail = source_inventory_left(src.id)
                    if avail < 10: continue
                    aim = world.plan_shot(src.id, last_planet.id, avail)
                    if aim and aim[1] < best_dist: best_dist, best_src = aim[1], src
                        
                if best_src and best_dist < world.remaining_steps - 2:
                    send = source_inventory_left(best_src.id)
                    aim = world.plan_shot(best_src.id, last_planet.id, send)
                    if aim:
                        append_move(best_src.id, aim[0], send)
                        spent_total[best_src.id] += send
                        planned_commitments[last_planet.id].append((aim[1], world.player, send))

    # --------------------------------------------------------
    # STAGE 1: COMET TROJAN AMBUSH
    # --------------------------------------------------------
    for cid, t_data in trojans.items():
        if expired(): break
        if t_data["strike_t"] <= 2:
            avail = source_inventory_left(cid)
            if avail < 15: continue
            aim = world.plan_shot(cid, t_data["target"].id, avail)
            if aim:
                angle, turns, _, _ = aim
                append_move(cid, angle, avail)
                spent_total[cid] += avail
                planned_commitments[t_data["target"].id].append((turns, world.player, avail))

    # --------------------------------------------------------
    # STAGE 2: VANGUARD GRAVITY MATRIX
    # --------------------------------------------------------
    global_operations = []
    
    for target in world.planets:
        if expired(): break
        op = calculate_target_gravity(target, world, planned_commitments, attack_budget, spent_total)
        if op: global_operations.append(op)

    global_operations.sort(key=lambda x: x["gravity"], reverse=True)

    # --------------------------------------------------------
    # STAGE 3: SWARM EXECUTION & HOLD FIRE SYNC
    # --------------------------------------------------------
    for op in global_operations:
        if expired(): break
        
        actual_accumulated = 0
        valid_participants = []
        
        for p in op['participants']:
            avail = max(0, attack_budget.get(p['src'].id, 0) - spent_total[p['src'].id])
            if avail >= 5:
                valid_participants.append({**p, 'current_avail': avail})
                actual_accumulated += avail
                
        if not valid_participants: continue
        
        max_turns = max(p['turns'] for p in valid_participants)
        sync_turn = max_turns + (3 if op['target_id'] in world.enemy_fights and world.is_four_player else 0)
        current_need = world.min_ships_to_own_by(op['target_id'], sync_turn, world.player, arrival_turn=sync_turn, planned_commitments=planned_commitments)
        
        if current_need <= 0 or actual_accumulated < current_need: continue
        
        target = world.planet_by_id[op['target_id']]
        
        # 🚨 DYNAMIC OVERKILL
        if target.owner == world.player: 
            remaining_need = min(actual_accumulated, current_need + 5) # Rescue
        elif target.owner != -1:
            if world.is_four_player: remaining_need = min(actual_accumulated, current_need + 10) 
            else: remaining_need = min(actual_accumulated, max(int(current_need * 1.5), 100))
        else:
            remaining_need = min(actual_accumulated, max(current_need + 5, 22)) 
            
        for p in valid_participants:
            if remaining_need <= 0: break
            send_amt = min(p['current_avail'], remaining_need)
            delay = sync_turn - p['turns']
            
            if delay > 0:
                spent_total[p['src'].id] += send_amt
                planned_commitments[op['target_id']].append((sync_turn, world.player, int(send_amt)))
            else:
                sent = append_move(p['src'].id, p['angle'], send_amt)
                if sent > 0:
                    spent_total[p['src'].id] += sent
                    planned_commitments[op['target_id']].append((p['turns'], world.player, int(sent)))
            
            remaining_need -= send_amt

    # --------------------------------------------------------
    # STAGE 4: LOGISTICAL SWEEP, VOID EVAC & TROJAN LOADING
    # --------------------------------------------------------
    frontline_planets = [p for p in world.my_planets if world.influence_map.get(p.id, 1e9) <= FRONT_LINE_DIST]
    
    loading_trojans = [world.planet_by_id[cid] for cid, data in trojans.items() if data["strike_t"] > 5 and data["life_left"] > 15]
    frontline_planets.extend(loading_trojans)
    
    if frontline_planets and not world.is_late:
        for core in world.my_planets:
            if expired(): break
            
            # 🚨 VOID EVACUATION: Dump everything from dying comets
            if core.id in world.comet_ids and world.comet_life(core.id) <= 3:
                left = int(core.ships) - spent_total[core.id]
                if left < 5: continue
                best_front = min([p for p in world.my_planets if p.id != core.id], key=lambda f: planet_distance_sq(core, f), default=None)
                if best_front:
                    aim = world.plan_shot(core.id, best_front.id, left)
                    if aim: append_move(core.id, aim[0], left); spent_total[core.id] += left
                continue

            if core in frontline_planets: continue 
            
            left = max(0, attack_budget.get(core.id, 0) - spent_total[core.id])
            if left < 10: continue
            
            safe_frontlines = [f for f in frontline_planets if world.base_timeline[f.id]["holds_full"] and f.id != core.id]
            if not safe_frontlines: safe_frontlines = [p for p in world.my_planets if p.id != core.id]
            if not safe_frontlines: continue
            
            best_front = max(safe_frontlines, key=lambda f: f.production - (dist(core.x, core.y, f.x, f.y)*0.5))
            aim = world.plan_shot(core.id, best_front.id, left)
            
            if aim:
                angle, turns, _, _ = aim
                append_move(core.id, angle, left)
                spent_total[core.id] += left
                planned_commitments[best_front.id].append((turns, world.player, left))

    return moves


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)

def build_world(obs):
    player = _read(obs, "player", 0)
    step = _read(obs, "step", 0) or 0
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    initial_planets = [Planet(*p) for p in raw_init]
    initial_by_id = {p.id: p for p in initial_planets}

    return WorldModel(
        player=player, step=step, planets=planets, fleets=fleets,
        initial_by_id=initial_by_id, ang_vel=ang_vel, comets=comets, comet_ids=comet_ids,
    )

def agent(obs, config=None):
    start_time = time.perf_counter()
    world = build_world(obs)
    if not world.my_planets: return []
    
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = min(SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
    deadline = start_time + soft_budget
    return plan_moves(world, deadline=deadline)

__all__ = ["agent", "build_world"]