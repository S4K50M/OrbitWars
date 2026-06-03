import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

# ============================================================
# Persistent Agent Memory
# ============================================================
GLOBAL_MEMORY = {
    "fsm_state": 0,
    "anvil_id": None
}

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
TOTAL_STEPS = 500
HORIZON = 200
ROUTE_SEARCH_HORIZON = 60
LAUNCH_CLEARANCE = 0.1
FLEET_SWEEP_HORIZON = 90

INTERCEPT_TOLERANCE = 1

INFLUENCE_DECAY = 1.2
BASE_DEFENSE_RATIO = 0.05

PRODUCTION_WEIGHT = 1.0
IMMEDIATE_SHIP_WEIGHT = 0.5
FRONTIER_MULTIPLIER = 1.4
CORE_MULTIPLIER = 0.8
DISTANCE_DECAY = 0.97

# Tunables from the rule-based opponent
MACRO_SHIP_COST_WEIGHT = 1.5
ATTACK_COST_TURN_WEIGHT = 0.85
TRAP_VULNERABLE_THRESHOLD = 5
TRAP_TRAVEL_LIMIT = 10
TRAP_GARRISON_MULT = 2.5
TRAP_PENALTY = 0.1
EXPOSED_OUTBOUND_RATIO = 0.7
EXPOSED_PLANET_VALUE_MULT = 3.0
EXPOSED_PLANET_SCORE_MULT = 3.0
BLOOD_IN_WATER_VALUE_MULT = 1.28
WEAKEST_ENEMY_VALUE_MULT = 1.12
ELIMINATION_BONUS = 18.0
WEAK_ENEMY_THRESHOLD = 45
DYNAMIC_DEFENSE_HIGH_RATIO = 0.85
DYNAMIC_DEFENSE_LOW_RATIO = 0.10
DYNAMIC_DEFENSE_THREAT_TURNS = 10
DYNAMIC_DEFENSE_THREAT_SHIPS = 50
DYNAMIC_DEFENSE_SAFE_TURNS = 25
PARTIAL_SOURCE_MIN_SHIPS = 6
EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
OPENING_FILTER_SAFE_PROD = 4
OPENING_FILTER_SAFE_TURNS = 10
FOUR_PLAYER_ROTATING_REACTION_GAP = 3
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.72
FOUR_PLAYER_ROTATING_TURN_LIMIT = 14
SNIPE_COST_TURN_WEIGHT = 0.45
MIN_ATTACK_SHIPS = 15   # never send a fleet smaller than this for captures/attacks

_log1000 = math.log(1000)
SPEED_CACHE = []
for i in range(1, 1001):
    SPEED_CACHE.append(1 + (MAX_SPEED - 1) * (math.log(i) / _log1000) ** 1.5)

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


@dataclass
class StrategicMission:
    target_id: int
    arrival_turn: int
    ships_needed: int
    priority_score: float
    mission_type: str


# ============================================================
# Physics (unchanged)
# ============================================================

def dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by))


def dist_sq(ax, ay, bx, by):
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by)


def orbital_radius(planet):
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def orbital_radius_sq(planet):
    return dist_sq(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static_planet(planet):
    return orbital_radius_sq(planet) >= (ROTATION_LIMIT - planet.radius) ** 2


def fleet_speed(ships):
    ships = int(ships)
    if ships <= 1:
        return 1.0
    elif ships >= 1000:
        return MAX_SPEED
    return SPEED_CACHE[ships - 1]


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
    return point_to_segment_distance_sq(CENTER_X, CENTER_Y, x1, y1, x2, y2) < (SUN_R + safety) ** 2


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
        sx, sy, sr, tx, ty, tr
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance


def _predict_planet_position_uncached(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r2 = dist_sq(init.x, init.y, CENTER_X, CENTER_Y)
    if r2 >= (ROTATION_LIMIT - init.radius) ** 2:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    r = math.sqrt(r2)
    return (CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang))


def predict_planet_position(planet, initial_by_id, angular_velocity, turns, cache=None):
    if cache is not None and planet.id in cache:
        pos_timeline = cache[planet.id]
        t = max(0, min(len(pos_timeline) - 1, turns))
        return pos_timeline[t]
    return _predict_planet_position_uncached(planet, initial_by_id, angular_velocity, turns)


def predict_comet_position(planet_id, comets, turns, cache=None):
    if cache is not None and planet_id in cache:
        path = cache[planet_id]
        t = max(0, min(len(path) - 1, turns))
        return path[t]
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


def predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids, cache=None):
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns, cache)
    return predict_planet_position(target, initial_by_id, ang_vel, turns, cache)


def target_can_move(target, initial_by_id, comet_ids):
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    r = dist_sq(init.x, init.y, CENTER_X, CENTER_Y)
    return r < (ROTATION_LIMIT - init.radius) ** 2


def search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids, cache=None):
    best = None
    best_score = None
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    tol = 0.1 if target_can_move(target, initial_by_id, comet_ids) else INTERCEPT_TOLERANCE

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target, candidate_turns, initial_by_id, ang_vel, comets, comet_ids, cache
        )
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
            target, actual_turns, initial_by_id, ang_vel, comets, comet_ids, cache
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x, src.y, src.radius, actual_pos[0], actual_pos[1], target.radius, ships
        )
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


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids, cache=None):
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        if not target_can_move(target, initial_by_id, comet_ids):
            return None
        return search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids, cache)

    tx, ty = target.x, target.y
    can_move = target_can_move(target, initial_by_id, comet_ids)
    tol = 0.1 if can_move else INTERCEPT_TOLERANCE

    for _ in range(5):
        _, turns = est
        pos = predict_target_position(target, turns, initial_by_id, ang_vel, comets, comet_ids, cache)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            if not can_move:
                return None
            return search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids, cache)
        if abs(ntx - tx) < 0.3 and abs(nty - ty) < 0.3 and abs(next_est[1] - turns) <= tol:
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids, cache)
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
            if dx * dx + dy * dy > max_dist * max_dist:
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
                pos = predict_target_position(
                    planet, turn, initial_by_id, ang_vel, comets, comet_ids
                )
                if pos is None:
                    continue
                if (
                    point_to_segment_distance(pos[0], pos[1], prev_x, prev_y, cur_x, cur_y)
                    <= planet.radius
                    and turn < best_time
                ):
                    best_time = turn
                    best_planet = planet
            prev_x = cur_x
            prev_y = cur_y

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def build_arrival_ledger(fleets, planets, initial_by_id=None, ang_vel=0.0, comets=(), comet_ids=()):
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(
            fleet, planets, initial_by_id=initial_by_id,
            ang_vel=ang_vel, comets=comets, comet_ids=comet_ids
        )
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
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


# ============================================================
# World Model
# ============================================================
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
        self.planet_by_id = {p.id: p for p in planets}

        self.my_planets = [p for p in planets if p.owner == player]
        self.enemy_planets = [p for p in planets if p.owner not in (-1, player)]
        self.neutral_planets = [p for p in planets if p.owner == -1]

        self.num_players = len({p.owner for p in planets if p.owner != -1} | {f.owner for f in fleets})
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_four_player = self.num_players > 2
        self.is_small_map = len(self.planets) <= 20
        self.is_very_late = self.remaining_steps < 25

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for p in planets:
            if p.owner != -1:
                self.owner_strength[p.owner] += p.ships
                self.owner_production[p.owner] += p.production
        for f in fleets:
            self.owner_strength[f.owner] += f.ships

        self.my_total = self.owner_strength[player]
        self.enemy_total = sum(s for o, s in self.owner_strength.items() if o != player)
        self.max_enemy_strength = max([s for o, s in self.owner_strength.items() if o != player], default=0)

        self.my_prod = self.owner_production[player]
        self.enemy_prod = sum(p for o, p in self.owner_production.items() if o != player)
        self.total_production = sum(p.production for p in planets)

        active_players = set(p.owner for p in planets if p.owner != -1) | {self.player}
        scores = {p: self.owner_strength[p] + (self.owner_production[p] * 50) for p in active_players}
        ranked = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
        self.leader = ranked[0] if ranked else self.player
        self.weakest = ranked[-1] if ranked else -1

        self.exposed_planet_ids = self._detect_exposed_enemy_planets()
        self.blood_in_water_owners = {
            owner for owner in self.owner_strength
            if owner != self.player and self.owner_strength[owner] <= WEAK_ENEMY_THRESHOLD
        }

        self.effective_horizon = min(HORIZON, self.remaining_steps)
        self._build_position_cache()
        self.arrivals = build_arrival_ledger(fleets, planets, initial_by_id, ang_vel, comets, comet_ids)
        self.timelines = {
            p.id: simulate_planet_timeline(p, self.arrivals.get(p.id, []), player, self.effective_horizon)
            for p in planets
        }
        self.influence_map = self._build_influence_map()

    def _detect_exposed_enemy_planets(self):
        exposed = set()
        for planet in self.enemy_planets:
            outbound = sum(
                int(f.ships) for f in self.fleets
                if f.owner == planet.owner and f.from_planet_id == planet.id and f.ships >= 5
            )
            total_visible = planet.ships + outbound
            if total_visible > 0 and outbound >= total_visible * EXPOSED_OUTBOUND_RATIO:
                exposed.add(planet.id)
        return exposed

    def _build_position_cache(self):
        self.pos_cache = {}
        for p in self.planets:
            if p.id in self.comet_ids:
                for group in self.comets:
                    pids = group.get("planet_ids", [])
                    if p.id in pids:
                        idx = pids.index(p.id)
                        paths = group.get("paths", [])
                        if idx < len(paths):
                            path = paths[idx]
                            start_idx = group.get("path_index", 0)
                            end_idx = min(len(path), start_idx + self.effective_horizon + 1)
                            self.pos_cache[p.id] = [(pt[0], pt[1]) for pt in path[start_idx:end_idx]]
                        break
            else:
                init = self.initial_by_id.get(p.id)
                if init is None:
                    self.pos_cache[p.id] = [(p.x, p.y)]
                    continue
                r2 = dist_sq(init.x, init.y, CENTER_X, CENTER_Y)
                if r2 >= (ROTATION_LIMIT - init.radius) ** 2:
                    self.pos_cache[p.id] = [(p.x, p.y)]
                    continue
                r = math.sqrt(r2)
                cur_ang = math.atan2(p.y - CENTER_Y, p.x - CENTER_X)
                path = []
                for t in range(self.effective_horizon + 1):
                    new_ang = cur_ang + self.ang_vel * t
                    path.append((CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)))
                self.pos_cache[p.id] = path

    def _build_influence_map(self):
        influence = defaultdict(float)
        for target in self.planets:
            inf = 0.0
            for src in self.planets:
                if src.id == target.id:
                    continue
                d2 = dist_sq(src.x, src.y, target.x, target.y)
                if d2 < 1:
                    continue
                power = src.production / (d2 ** (INFLUENCE_DECAY / 2))
                if src.owner == self.player:
                    inf += power
                elif src.owner != -1:
                    inf -= power
            for f in self.fleets:
                d2 = dist_sq(f.x, f.y, target.x, target.y)
                if d2 < 1:
                    continue
                fleet_power = (f.ships / 50.0) / (d2 ** (INFLUENCE_DECAY / 2))
                if f.owner == self.player:
                    inf += fleet_power
                else:
                    inf -= fleet_power
            influence[target.id] = inf
        return influence

    def state_at(self, target_id, turns):
        tl = self.timelines[target_id]
        t = min(max(0, int(turns)), tl["horizon"])
        return tl["owner_at"].get(t, -1), tl["ships_at"].get(t, 0)

    def dynamic_defense_keep(self, planet):
        min_enemy_time = 1e9
        nearest_enemy_ships = 0
        for enemy in self.enemy_planets:
            t = travel_time(planet.x, planet.y, planet.radius, enemy.x, enemy.y, enemy.radius, 1)
            if t < min_enemy_time:
                min_enemy_time = t
                nearest_enemy_ships = int(enemy.ships)

        if min_enemy_time <= DYNAMIC_DEFENSE_THREAT_TURNS and nearest_enemy_ships > DYNAMIC_DEFENSE_THREAT_SHIPS:
            ratio = DYNAMIC_DEFENSE_HIGH_RATIO
        elif min_enemy_time > DYNAMIC_DEFENSE_SAFE_TURNS:
            ratio = DYNAMIC_DEFENSE_LOW_RATIO
        else:
            progress = (min_enemy_time - DYNAMIC_DEFENSE_THREAT_TURNS) / max(1, DYNAMIC_DEFENSE_SAFE_TURNS - DYNAMIC_DEFENSE_THREAT_TURNS)
            progress = max(0.0, min(1.0, progress))
            ratio = DYNAMIC_DEFENSE_LOW_RATIO + (DYNAMIC_DEFENSE_HIGH_RATIO - DYNAMIC_DEFENSE_LOW_RATIO) * (1.0 - progress)

        stacked_threat = 0
        enemy_etas = []
        for enemy in self.enemy_planets:
            t = travel_time(enemy.x, enemy.y, enemy.radius, planet.x, planet.y, planet.radius, 1)
            if t <= 14:
                enemy_etas.append((t, int(enemy.ships)))
        enemy_etas.sort()
        for i in range(len(enemy_etas)):
            window_sum = enemy_etas[i][1]
            for j in range(i + 1, len(enemy_etas)):
                if enemy_etas[j][0] - enemy_etas[i][0] <= 3:
                    window_sum += enemy_etas[j][1]
            stacked_threat = max(stacked_threat, window_sum)
        proactive = int(stacked_threat * ratio * 0.22)
        return proactive

    def reaction_times(self, target_id):
        my_best = 1e9
        enemy_best = 1e9
        for planet in self.my_planets:
            aim = aim_with_prediction(planet, self.planet_by_id[target_id],
                                      max(1, int(planet.ships)),
                                      self.initial_by_id, self.ang_vel,
                                      self.comets, self.comet_ids, self.pos_cache)
            if aim:
                my_best = min(my_best, aim[1])
        for planet in self.enemy_planets:
            aim = aim_with_prediction(planet, self.planet_by_id[target_id],
                                      max(1, int(planet.ships)),
                                      self.initial_by_id, self.ang_vel,
                                      self.comets, self.comet_ids, self.pos_cache)
            if aim:
                enemy_best = min(enemy_best, aim[1])
        return my_best, enemy_best

    def min_ships_to_own_at(self, target_id, arrival_turn, attacker_owner,
                            planned_commitments=None, extra_arrivals=(), upper_bound=None):
        planned_commitments = planned_commitments or {}
        arrival_turn = max(1, int(math.ceil(arrival_turn)))

        def owns_at(ships):
            extra = list(extra_arrivals) + [(arrival_turn, attacker_owner, int(ships))]
            timeline = self.projected_timeline(target_id, arrival_turn,
                                               planned_commitments=planned_commitments,
                                               extra_arrivals=extra)
            return timeline["owner_at"].get(arrival_turn) == attacker_owner

        lo, hi = 1, upper_bound or 10000
        if not owns_at(hi):
            return hi + 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def projected_timeline(self, target_id, horizon, planned_commitments=None, extra_arrivals=()):
        planned_commitments = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [item for item in self.arrivals.get(target_id, []) if item[0] <= horizon]
        arrivals.extend(item for item in planned_commitments.get(target_id, []) if item[0] <= horizon)
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        target = self.planet_by_id[target_id]
        return simulate_planet_timeline(target, arrivals, self.player, horizon)


# ============================================================
# TOT parameters
# ============================================================
def calculate_ships_for_speed(target_speed):
    if target_speed <= 1.0:
        return 1
    if target_speed >= MAX_SPEED:
        return 1000
    ratio = ((target_speed - 1.0) / (MAX_SPEED - 1.0)) ** (2.0 / 3.0)
    return int(math.ceil(math.exp(ratio * _log1000)))


def get_tot_launch_parameters(src, target, target_turns, world):
    pos = predict_target_position(target, target_turns, world.initial_by_id, world.ang_vel,
                                  world.comets, world.comet_ids, world.pos_cache)
    if not pos:
        return None
    tx, ty = pos
    safe = safe_angle_and_distance(src.x, src.y, src.radius, tx, ty, target.radius)
    if not safe:
        return None
    angle, dist_to_hit = safe
    req_speed = dist_to_hit / max(1, target_turns)
    if req_speed > MAX_SPEED:
        return None
    min_ships = calculate_ships_for_speed(req_speed)
    speed_too_fast = dist_to_hit / max(1, target_turns - 1)
    max_ships = calculate_ships_for_speed(speed_too_fast) - 1
    if max_ships < min_ships:
        max_ships = min_ships
    if max_ships > 1000:
        max_ships = 999999
    return min_ships, max_ships, angle


# ============================================================
# Opening filter
# ============================================================
def opening_filter(target, arrival_turns, needed, src_available, world):
    if world.step >= OPENING_TURN_LIMIT or target.owner != -1:
        return False
    if target.id in world.comet_ids:
        return False
    if is_static_planet(target):
        return False
    my_t, enemy_t = world.reaction_times(target.id)
    reaction_gap = enemy_t - my_t
    if (target.production >= OPENING_FILTER_SAFE_PROD and
        arrival_turns <= OPENING_FILTER_SAFE_TURNS and
        reaction_gap >= SAFE_NEUTRAL_MARGIN):
        return False
    if world.is_four_player:
        affordable = needed <= max(PARTIAL_SOURCE_MIN_SHIPS, int(src_available * FOUR_PLAYER_ROTATING_SEND_RATIO))
        if (affordable and arrival_turns <= FOUR_PLAYER_ROTATING_TURN_LIMIT and
            reaction_gap >= FOUR_PLAYER_ROTATING_REACTION_GAP):
            return False
        return True
    return arrival_turns > 13 or target.production <= 2


# ============================================================
# Snipe missions
# ============================================================
def build_snipe_missions(world):
    """Return a list of StrategicMission snipe missions."""
    missions = []
    for target in world.neutral_planets:
        enemy_events = [
            (int(math.ceil(eta)), owner, int(ships))
            for eta, owner, ships in world.arrivals.get(target.id, [])
            if owner not in (-1, world.player) and ships > 0
        ]
        if not enemy_events:
            continue
        enemy_events.sort()
        for enemy_eta, _, _ in enemy_events:
            if enemy_eta > world.effective_horizon:
                continue
            desired_turn = min(world.effective_horizon, enemy_eta + 1)
            owner_at, _ = world.state_at(target.id, desired_turn)
            if owner_at != -1:
                continue
            for src in world.my_planets:
                src_available = world.planet_by_id[src.id].ships
                needed = world.min_ships_to_own_at(target.id, desired_turn, world.player,
                                                   planned_commitments={},
                                                   upper_bound=src_available)
                if needed <= 0 or needed > src_available:
                    continue
                aim = aim_with_prediction(src, target, needed, world.initial_by_id,
                                          world.ang_vel, world.comets, world.comet_ids,
                                          world.pos_cache)
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                if turns > desired_turn:
                    continue
                score = target.production * (world.remaining_steps - turns) / (needed + turns * SNIPE_COST_TURN_WEIGHT + 1.0)
                # Create StrategicMission instead of Mission
                missions.append(StrategicMission(
                    target_id=target.id,
                    arrival_turn=desired_turn,
                    ships_needed=needed,
                    priority_score=score,
                    mission_type="snipe"
                ))
                break
    return missions

# ============================================================
# Strategic mission generation helpers
# ============================================================
def is_trap(src, target, send_cap, world):
    if target.ships >= TRAP_VULNERABLE_THRESHOLD:
        return False
    for enemy in world.enemy_planets:
        if enemy.id == target.id or enemy.owner != target.owner:
            continue
        t = travel_time(src.x, src.y, src.radius, enemy.x, enemy.y, enemy.radius, 1)
        if t <= TRAP_TRAVEL_LIMIT and enemy.ships > send_cap * TRAP_GARRISON_MULT:
            return True
    return False


def is_economically_viable(value, send_cap, turns):
    cost = send_cap * MACRO_SHIP_COST_WEIGHT
    return value > cost


def target_value(target, arrival_turns, mission, world, indirect_wealth_map=None):
    turns_profit = max(1, world.remaining_steps - arrival_turns)
    if target.id in world.comet_ids:
        life = comet_remaining_life(target.id, world.comets)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit
    if indirect_wealth_map:
        friendly, neutral, enemy = indirect_wealth_map.get(target.id, (0, 0, 0))
        value += (friendly * 0.35 + neutral * 0.9 + enemy * 1.25) * turns_profit * 0.15

    if is_static_planet(target):
        value *= 1.4 if target.owner == -1 else 1.6

    if world.is_four_player:
        if target.id in world.exposed_planet_ids:
            value *= EXPOSED_PLANET_VALUE_MULT
        if target.owner in world.blood_in_water_owners:
            value *= BLOOD_IN_WATER_VALUE_MULT
            value += ELIMINATION_BONUS * 0.6
        if target.owner == world.weakest and target.owner not in (-1, world.player):
            value *= WEAKEST_ENEMY_VALUE_MULT

    if world.remaining_steps < 60:
        value += max(0, target.ships) * 0.6
        if target.owner not in (-1, world.player) and world.owner_strength.get(target.owner, 0) <= WEAK_ENEMY_THRESHOLD:
            value += ELIMINATION_BONUS

    return value


def original_generate_strategic_missions(world):
    missions = []
    eff_hor = world.effective_horizon

    indirect_map = {}
    for target in world.planets:
        friendly = 0.0
        neutral = 0.0
        enemy = 0.0
        for other in world.planets:
            if other.id == target.id:
                continue
            d = dist(target.x, target.y, other.x, other.y)
            if d < 1:
                continue
            factor = other.production / (d + 5.0)
            if other.owner == world.player:
                friendly += factor
            elif other.owner == -1:
                neutral += factor
            else:
                enemy += factor
        indirect_map[target.id] = (friendly, neutral, enemy)

    for target in world.planets:
        target_id = target.id
        timeline = world.timelines[target_id]

        if target.owner == world.player:
            if not timeline["holds_full"] and timeline["fall_turn"] is not None:
                fall_turn = timeline["fall_turn"]
                needed = timeline["keep_needed"] + 5
                missions.append(StrategicMission(
                    target_id, max(1, fall_turn - 1), needed, 10000.0, "defense"
                ))
            continue

        if target.owner != world.player:
            for t in range(1, eff_hor + 1):
                owner_t, garrison_t = world.state_at(target_id, t)
                if owner_t not in (-1, world.player):
                    vulture_turn = min(eff_hor, t + 2)
                    if vulture_turn <= eff_hor:
                        owner_v, garrison_v = world.state_at(target_id, vulture_turn)
                        if owner_v != world.player:
                            needed = int(garrison_v) + 2
                            score = (target.production * 500.0) / max(1, needed + vulture_turn)
                            missions.append(StrategicMission(
                                target_id, vulture_turn, needed, score, "vulture"
                            ))
                    break

        is_neutral_or_comet = (target.owner == -1) or (target_id in world.comet_ids)
        force_ratio = world.my_total / max(1, world.enemy_total)
        can_attack_enemy = (force_ratio > 1.2) or (world.step > 70)

        if not is_neutral_or_comet and not can_attack_enemy:
            continue

        if target_id in world.comet_ids:
            life = comet_remaining_life(target_id, world.comets)
            if life == 0:
                continue

        scan_step = 2 if world.is_small_map else 3
        for turns_ahead in range(1, eff_hor + 1, scan_step):
            if target_id in world.comet_ids and turns_ahead >= comet_remaining_life(target_id, world.comets):
                continue

            owner_at, garrison_at = world.state_at(target_id, turns_ahead)
            if owner_at == world.player:
                continue

            if not is_neutral_or_comet:
                inf = world.influence_map.get(target_id, 0.0)
                if inf < -3.0 and world.step < 100:
                    continue

            needed = int(garrison_at) + 2
            if not is_neutral_or_comet:
                shadow = min(150, int(world.owner_strength.get(owner_at, 0) * 0.25))
                needed += shadow + 5

            val = target_value(target, turns_ahead, "capture", world, indirect_map)
            if val <= 0:
                continue
            if not is_economically_viable(val, needed, turns_ahead):
                continue

            time_penalty = DISTANCE_DECAY ** turns_ahead
            score = (val / (needed + turns_ahead * ATTACK_COST_TURN_WEIGHT + 1.0)) * time_penalty

            for enemy in world.enemy_planets:
                if enemy.owner == target.owner and enemy.id != target_id:
                    t = travel_time(enemy.x, enemy.y, enemy.radius, target.x, target.y, target.radius, 1)
                    if t <= TRAP_TRAVEL_LIMIT and enemy.ships > needed * TRAP_GARRISON_MULT:
                        score *= TRAP_PENALTY
                        break

            missions.append(StrategicMission(
                target_id, turns_ahead, needed, score,
                "expand" if is_neutral_or_comet else "attack"
            ))

    return missions


def generate_strategic_missions(world):
    missions = original_generate_strategic_missions(world)
    missions.extend(build_snipe_missions(world))
    missions.sort(key=lambda m: m.priority_score, reverse=True)
    return missions

# ============================================================
# TOT swarm execution (with tracking)
# ============================================================
def execute_tot_swarms_capture_tracking(world, missions, available_surplus, deadline=None):
    moves = []
    claimed_targets = set()

    my_prod_ratio = world.my_prod / max(1, world.total_production)
    is_dominating = (not world.is_four_player and my_prod_ratio > 0.55) or \
                    (world.is_four_player and my_prod_ratio > 0.30)

    for mission in missions:
        if deadline and time.perf_counter() > deadline:
            break
        if mission.priority_score < 0 or mission.target_id in claimed_targets:
            continue

        target = world.planet_by_id[mission.target_id]
        target_turn = mission.arrival_turn
        base_needed = mission.ships_needed

        if is_dominating and target.owner not in (-1, world.player):
            actual_needed = base_needed * 2
        else:
            actual_needed = base_needed

        potential_contributors = []
        for src_id, surplus in available_surplus.items():
            if surplus < 5 or src_id == target.id:
                continue
            src = world.planet_by_id[src_id]
            tot_params = get_tot_launch_parameters(src, target, target_turn, world)
            if tot_params is None:
                continue
            min_req, max_req, angle = tot_params
            if min_req > surplus:
                continue
            effective_max = min(max_req, surplus)

            if base_needed >= 15 and effective_max < 10:
                continue
            actual_min = max(min_req, 10) if max_req >= 10 else min_req

            potential_contributors.append({
                "src_id": src_id,
                "min_req": actual_min,
                "max_req": effective_max,
                "angle": angle
            })

        potential_contributors.sort(key=lambda c: c["max_req"], reverse=True)

        current_package_force = 0
        committed_this_mission = []

        for contrib in potential_contributors:
            if current_package_force >= actual_needed:
                break
            send_amt = contrib["min_req"]
            if current_package_force + send_amt < actual_needed:
                shortfall = actual_needed - (current_package_force + send_amt)
                extra = min(shortfall, contrib["max_req"] - send_amt)
                send_amt += extra
            current_package_force += send_amt
            committed_this_mission.append((contrib["src_id"], contrib["angle"], send_amt))

        if current_package_force >= base_needed:
            for src_id, angle, send_amt in committed_this_mission:
                moves.append([src_id, float(angle), int(send_amt)])
                available_surplus[src_id] -= send_amt
            claimed_targets.add(mission.target_id)

    return moves, claimed_targets


# ============================================================
# Staging (unchanged)
# ============================================================
def execute_staging(world, available_surplus):
    moves = []
    frontline = [p for p in world.my_planets if world.influence_map.get(p.id, 0.0) <= 2.0]
    if not frontline:
        frontline = world.my_planets
    for src_id, surplus in available_surplus.items():
        if surplus < 30:
            continue
        src = world.planet_by_id[src_id]
        inf = world.influence_map.get(src_id, 0.0)
        if inf > 3.5:
            best_front = min(
                frontline,
                key=lambda p: dist_sq(src.x, src.y, p.x, p.y) + (world.influence_map.get(p.id, 0.0) * 1000)
            )
            if best_front.id == src.id:
                continue
            aim = aim_with_prediction(
                src, best_front, surplus, world.initial_by_id,
                world.ang_vel, world.comets, world.comet_ids, world.pos_cache
            )
            if aim:
                angle, _, _, _ = aim
                moves.append([src.id, float(angle), int(surplus)])
                available_surplus[src.id] -= surplus
    return moves


# ============================================================
# Greedy second pass
# ============================================================
def greedy_second_pass(world, available_surplus):
    extra_moves = []
    for p in world.my_planets:
        surplus = available_surplus.get(p.id, 0)
        if surplus < MIN_ATTACK_SHIPS:          # new minimum
            continue
        best_score = -1
        best_plan = None
        for target in world.planets:
            if target.owner == world.player or target.id == p.id:
                continue
            if target.id in world.comet_ids:
                life = comet_remaining_life(target.id, world.comets)
                if life == 0:
                    continue
            est_turns = travel_time(p.x, p.y, p.radius, target.x, target.y, target.radius, 1)
            if est_turns > world.effective_horizon:
                continue
            needed = world.min_ships_to_own_at(target.id, est_turns, world.player,
                                               planned_commitments={},
                                               upper_bound=surplus)
            if needed <= 0 or needed > surplus:
                continue
            needed = max(needed, MIN_ATTACK_SHIPS)   # enforce minimum
            if needed > surplus:
                continue
            if opening_filter(target, est_turns, needed, surplus, world):
                continue
            aim = aim_with_prediction(p, target, needed, world.initial_by_id,
                                      world.ang_vel, world.comets, world.comet_ids,
                                      world.pos_cache)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            val = target_value(target, turns, "capture", world, None)
            if val <= 0:
                continue
            score = val / (turns + 1)
            if score > best_score:
                best_score = score
                best_plan = (target.id, angle, turns, needed)
        if best_plan:
            tid, angle, turns, needed = best_plan
            extra_moves.append([p.id, float(angle), int(needed)])
            available_surplus[p.id] -= needed
    return extra_moves


# ============================================================
# Main planning function
# ============================================================
def plan_moves(world, deadline=None):
    available_surplus = {}
    for p in world.my_planets:
        exact_keep = world.timelines[p.id]["keep_needed"]
        dynamic = world.dynamic_defense_keep(p)
        if world.step < OPENING_TURN_LIMIT and len(world.my_planets) <= 3:
            dynamic = int(dynamic * 0.2)
            total_keep = max(exact_keep, dynamic)
            total_keep = min(total_keep, int(p.ships * 0.5))
        else:
            reserve = max(exact_keep, dynamic)
            threat = max(0, -world.influence_map.get(p.id, 0))
            buffer = int(threat * 1.5 + p.ships * BASE_DEFENSE_RATIO)
            total_keep = reserve + buffer
        available_surplus[p.id] = max(0, int(p.ships) - total_keep)

    # 1. Immediate captures
    immediate_moves = []
    for p in world.my_planets:
        surplus = available_surplus.get(p.id, 0)
        if surplus < MIN_ATTACK_SHIPS:
            continue
        best_target = None
        best_score = -1
        for n in sorted(world.neutral_planets, key=lambda n: dist_sq(p.x, p.y, n.x, n.y)):
            future_owner, _ = world.state_at(n.id, 15)
            if future_owner not in (-1, world.player):
                continue
            need = max(int(n.ships) + 2, MIN_ATTACK_SHIPS)
            if surplus < need:
                continue
            if opening_filter(n, 0, need, surplus, world):
                continue
            aim = aim_with_prediction(p, n, need, world.initial_by_id,
                                      world.ang_vel, world.comets, world.comet_ids,
                                      world.pos_cache)
            if aim is None:
                continue
            angle, turns, _, _ = aim
            val = n.production * (world.remaining_steps - turns)
            if val <= 0:
                continue
            score = val / (turns + 1)
            if score > best_score:
                best_score = score
                best_target = (n.id, angle, turns, need)
        if best_target:
            tid, angle, turns, need = best_target
            immediate_moves.append([p.id, float(angle), int(need)])
            available_surplus[p.id] -= need

    # 2. TOT swarms + snipes
    missions = generate_strategic_missions(world)
    swarm_moves, claimed = execute_tot_swarms_capture_tracking(world, missions, available_surplus, deadline)

    offensive_moves = immediate_moves + swarm_moves

    # 3. Staging (only if attacking)
    staging_moves = []
    if offensive_moves:
        staging_moves = execute_staging(world, available_surplus)

    # 4. Greedy second pass
    if not world.is_very_late:
        greedy_moves = greedy_second_pass(world, available_surplus)
        offensive_moves.extend(greedy_moves)

    # 5. Fallback forced capture
    if not offensive_moves and world.neutral_planets:
        for p in world.my_planets:
            surplus = available_surplus.get(p.id, 0)
            if surplus < MIN_ATTACK_SHIPS:
                continue
            for n in sorted(world.neutral_planets, key=lambda n: dist_sq(p.x, p.y, n.x, n.y)):
                future_owner, _ = world.state_at(n.id, 15)
                if future_owner not in (-1, world.player):
                    continue
                need = max(int(n.ships) + 2, MIN_ATTACK_SHIPS)
                if surplus < need:
                    continue
                aim = aim_with_prediction(p, n, need, world.initial_by_id,
                                          world.ang_vel, world.comets, world.comet_ids,
                                          world.pos_cache)
                if aim is None:
                    continue
                angle, turns, _, _ = aim
                offensive_moves.append([p.id, float(angle), int(need)])
                available_surplus[p.id] -= need
                break

    return offensive_moves + staging_moves


# ============================================================
# Agent entry point
# ============================================================
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
        initial_by_id=initial_by_id, ang_vel=ang_vel,
        comets=comets, comet_ids=comet_ids
    )


def agent(obs, config=None):
    start_time = time.perf_counter()
    world = build_world(obs)
    if not world.my_planets:
        return []
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    deadline = start_time + min(0.82, act_timeout * 0.82)
    moves = plan_moves(world, deadline=deadline)
    if world.step % 10 == 0:
        my_p, tot_p = world.my_prod, world.total_production
        ratio = my_p / max(1, tot_p)
        status = "👑 DOMINATING" if ratio > 0.45 else ("⚠️ BEHIND" if ratio < 0.2 else "⚔️ CONTESTED")
        print(f"\n[TICK {world.step:03d} | P{world.player} | {status}] 🏭 Econ: {my_p}/{tot_p} ({ratio*100:.1f}%) | 🚀 Fleet: {world.my_total} vs Max Enemy: {world.max_enemy_strength}")
        print(f"📡 Strike Packages Launched: {len(moves)} {'='*40}", flush=True)
    return moves


__all__ = ["agent", "build_world"]