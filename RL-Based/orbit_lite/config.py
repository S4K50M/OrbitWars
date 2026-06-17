from .movement import MovementConfig

from dataclasses import dataclass
from .constants import DEFAULT_MOVEMENT_HORIZON, DEFAULT_DRIFT_EPSILON, DEFAULT_MAX_TRACKED_FLEETS
import dataclasses




def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:

    """MovementConfig: fleet tracking on, horizon = config.horizon."""

    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


@dataclass(frozen=True)
class ProducerLiteConfig:

    """Behaviour knobs.  """
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18

    # --- shortlists ------------------------------------------------------

    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          

# --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              
    min_ships_to_launch: float = 4.0
    # --- regroup  ------------------------------

    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3

    # --- potential-attack RISK_SCORE (added to the regroup gradient) -----

    enable_potential_risk: bool = False     # defense feature off by default (no measured gain)
    risk_blend_weight: float = 1.0          # weight of RISK added to cheap_enemy_pressure
    risk_enemy_prod_weight: float = 2.0     # enemy strength = ships + w * production
    risk_self_prod_weight: float = 2.0      # own-planet value  = 1 + w * production
    risk_support_weight: float = 0.5        # friendly-neighbour discount strength

    # --- focus fire (coordinated multi-source attack) -------------------

    enable_focus_fire: bool = True          # pool same-step sources to take strong targets
    max_strike_sources: int = 4             # max planets combined per strike (contributor cap L)

    # --- Voronoi Territorial Control ---
    enable_voronoi_features: bool = True
    voronoi_deep_enemy_penalty: float = 0.6  # Weight for penalizing attacks deep in enemy space
    voronoi_border_focus: float = 0.4        # Weight for pulling regroup ships to the border

    enable_roi_targeting: bool = True
    roi_production_weight: float = 15.0      # Priority weight for high-yield planets

    # --- NEW: Fractional Sweep (Replaces Beam Search) ---
    enable_fractional_sizes: bool = False


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    risk_blend_weight=0.5,                   # damp the precaution in diffuse 4p FFA
    max_strike_sources=3,                    # leaner combined strikes in 4p FFA

    enable_voronoi_features = True,
    voronoi_deep_enemy_penalty = 3.0,
    voronoi_border_focus = 2.0,
    roi_production_weight = 10.0,

    enable_fractional_sizes = True
)

class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
