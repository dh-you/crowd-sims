import os
import random
import sys
from dataclasses import dataclass

import numpy as np
from pygame.math import Vector2

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.agent import Agent
from core.physics import advance_velocity_agents
from core.wall import Wall


CONFIG = {
    "COUNT": 102,
    "RADIUS": 3,
    "MAXSPEED": 5,
    "MAXFORCE": 30,
    "HORIZON": 5,
    "K": 2,
    "AVOID": 10,
    "SIDESTEP": 5,
}


@dataclass(frozen=True)
class CabinLayout:
    """Single-aisle seat geometry with a fixed number of rows."""
    left_seats: int
    right_seats: int
    rows: int = 17

    def __post_init__(self):
        if self.left_seats < 1 or self.right_seats < 1 or self.rows < 1:
            raise ValueError("layout seats and rows must be positive")
        if max(self.left_seats, self.right_seats) > 4:
            raise ValueError("single-aisle layouts support at most four seats per side")

    @property
    def capacity(self):
        return (self.left_seats + self.right_seats) * self.rows

    @property
    def label(self):
        return f"{self.left_seats}x{self.right_seats}"

    @property
    def seat_columns(self):
        right = [4 * index for index in range(self.right_seats, 0, -1)]
        left = [-4 * index for index in range(self.left_seats, 0, -1)]
        return right + left

    @property
    def outer_wall_column(self):
        return 4 * max(self.left_seats, self.right_seats) + 2


@dataclass(frozen=True)
class WidebodyLayout:
    """Two-aisle layout represented as left, middle, and right seat blocks."""
    left_seats: int = 3
    middle_seats: int = 4
    right_seats: int = 3
    rows: int = 17

    def __post_init__(self):
        if min(self.left_seats, self.middle_seats, self.right_seats, self.rows) < 1:
            raise ValueError("layout seats and rows must be positive")
        if self.middle_seats % 2:
            raise ValueError("two-aisle middle seat block must be even")
        if (self.left_seats, self.middle_seats, self.right_seats) != (3, 4, 3):
            raise ValueError("the supported two-aisle layout is 3x4x3")

    @property
    def capacity(self):
        return (self.left_seats + self.middle_seats + self.right_seats) * self.rows

    @property
    def label(self):
        return f"{self.left_seats}x{self.middle_seats}x{self.right_seats}"

    @property
    def outer_wall_column(self):
        return 24

    @property
    def aisle_columns(self):
        return (-10, 10)

    @property
    def seat_specs(self):
        # (seat column, assigned aisle index, assigned aisle column)
        return (
            [(-22, 0, -10), (-18, 0, -10), (-14, 0, -10)]
            + [(-6, 0, -10), (-2, 0, -10)]
            + [(2, 1, 10), (6, 1, 10)]
            + [(22, 1, 10), (18, 1, 10), (14, 1, 10)]
        )


DEFAULT_LAYOUT = CabinLayout(3, 3)


def parse_layout(layout):
    if isinstance(layout, (CabinLayout, WidebodyLayout)):
        return layout
    if layout is None:
        return DEFAULT_LAYOUT
    try:
        blocks = tuple(int(value) for value in str(layout).lower().split("x"))
    except ValueError as error:
        raise ValueError(f"layout must look like '3x3' or '3x4x3', got {layout!r}") from error
    if len(blocks) == 2:
        return CabinLayout(*blocks)
    if len(blocks) == 3:
        return WidebodyLayout(*blocks)
    raise ValueError(f"layout must look like '3x3' or '3x4x3', got {layout!r}")

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
TIMESTEP = 0.05
TRAINING_EPISODE_CAP = 20000

STATE_DIM = 4
ACTION_DIM = 2
STATE_NAMES = (
    "aisle_offset",
    "row_clearance",
    "opposing_clearance",
    "forward_clearance",
)
POLICY_HIDDEN_DIMS = (32, 32)
POLICY_STOP_THRESHOLD = 0.25
LATERAL_POLICY_STOP_THRESHOLD = 2.5
POLICY_CROSS_AXIS_DEADBAND = 0.5
ACTION_REGIME_THRESHOLD = 0.25
STATE_IS_NORMALIZED = True
BALANCE_ACTION_REGIMES = True
SYNTHETIC_SAMPLES_PER_REGIME = 30000
SYNTHETIC_BOUNDARY_SAMPLES_PER_SIDE = 5000
SYNTHETIC_BOTTOM_ROW_REPEATS = 2000
# A premature release is less acceptable than a delayed release at a local
# clearance threshold.  train_bc.py reads this scene-specific default.
WAITING_WEIGHT = 10.0
CROSS_AXIS_WEIGHT = 50.0

AISLE_X = 300.0
EXIT_Y = 420.0
# Fixed across every supported layout: a fourth single-aisle seat is 48 px
# from its aisle. This avoids layout-dependent feature meanings.
MAX_SEAT_DISTANCE = 48.0
SENSOR_RANGE = 100.0
AISLE_OCCUPANCY_HALF_WIDTH = 6.0
AISLE_CENTER_EPS = 2.0
# Compatibility aliases used by the lean evaluator and older tests.
AISLE_HALF_WIDTH = AISLE_OCCUPANCY_HALF_WIDTH
FORWARD_CORRIDOR_HALF_WIDTH = 15.0
FORWARD_MIN_Y = 6.0

AISLE_OCCUPANCY_STATE_EPS = AISLE_OCCUPANCY_HALF_WIDTH / MAX_SEAT_DISTANCE
AISLE_CENTER_STATE_EPS = AISLE_CENTER_EPS / MAX_SEAT_DISTANCE
AISLE_EPS = AISLE_OCCUPANCY_STATE_EPS
ROW_SAFE = 15.0 / SENSOR_RANGE
FOLLOW_STOP = 24.0 / SENSOR_RANGE
FOLLOW_FULL = 40.0 / SENSOR_RANGE
FOLLOW_SAFE = FOLLOW_STOP
MERGE_SAFE = 30.0 / SENSOR_RANGE

PREFERRED_SPEED_MIN = 4.0
PREFERRED_SPEED_MAX = 6.0


def project_state_observations(states, omitted_name=None):
    """Return full observations or a named three-input ablation projection."""
    values = np.asarray(states)
    if omitted_name is None:
        return values
    try:
        omitted_index = STATE_NAMES.index(omitted_name)
    except ValueError as error:
        choices = ", ".join(STATE_NAMES)
        raise ValueError(
            f"unknown airplane observation {omitted_name!r}; choose from {choices}"
        ) from error
    if values.shape[-1] != len(STATE_NAMES):
        raise ValueError(
            f"expected {len(STATE_NAMES)} airplane observations, got {values.shape[-1]}"
        )
    return np.delete(values, omitted_index, axis=-1)


def extract_state(agent, agents):
    """Return four normalized local observations for autonomous deboarding."""
    aisle_x = agent.getData("aisle_x")
    if aisle_x is None:
        aisle_x = AISLE_X
    aisle_delta = aisle_x - agent.position.x
    aisle_offset = np.clip(
        aisle_delta / MAX_SEAT_DISTANCE,
        -1.0,
        1.0,
    )

    start_row = agent.getData("start_row")
    start_side = agent.getData("start_side")
    start_aisle = agent.getData("start_aisle")
    if start_aisle is None:
        start_aisle = 0
    row_clearance = SENSOR_RANGE
    opposing_clearance = SENSOR_RANGE
    forward_clearance = SENSOR_RANGE

    direction_to_aisle = np.sign(aisle_delta)
    distance_to_aisle = abs(aisle_delta)
    occupies_aisle = distance_to_aisle <= AISLE_OCCUPANCY_HALF_WIDTH + 1e-6

    for other in agents:
        if other is agent or not filter_agent(other):
            continue

        other_row = other.getData("start_row")
        other_side = other.getData("start_side")
        other_aisle = other.getData("start_aisle")
        if other_aisle is None:
            other_aisle = 0
        distance = agent.position.distance_to(other.position)

        if other_row == start_row and other_aisle == start_aisle:
            if other_side == start_side and direction_to_aisle != 0:
                lateral_progress = (
                    other.position.x - agent.position.x
                ) * direction_to_aisle
                if 0.0 < lateral_progress <= distance_to_aisle + other.radius:
                    row_clearance = min(row_clearance, distance)
            elif other_side != start_side:
                opposing_clearance = min(opposing_clearance, distance)

        y_diff = other.position.y - agent.position.y
        if (
            (other_row != start_row or occupies_aisle)
            and other_aisle == start_aisle
            and y_diff > FORWARD_MIN_Y
            and abs(other.position.x - aisle_x)
            <= FORWARD_CORRIDOR_HALF_WIDTH
        ):
            forward_clearance = min(forward_clearance, y_diff)

    return np.array(
        [
            aisle_offset,
            min(row_clearance, SENSOR_RANGE) / SENSOR_RANGE,
            min(opposing_clearance, SENSOR_RANGE) / SENSOR_RANGE,
            min(forward_clearance, SENSOR_RANGE) / SENSOR_RANGE,
        ],
        dtype=np.float32,
    )


def expert_action(state):
    """Map only the four observable values to an expert velocity."""
    aisle_offset, row_clearance, opposing_clearance, forward_clearance = state

    if abs(float(aisle_offset)) > AISLE_CENTER_STATE_EPS + 1e-6:
        blocked = (
            row_clearance < ROW_SAFE
            or forward_clearance < FOLLOW_STOP
            or (aisle_offset < 0 and opposing_clearance < MERGE_SAFE)
        )
        if blocked:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        return np.array(
            [np.sign(aisle_offset) * CONFIG["MAXSPEED"], 0.0],
            dtype=np.float32,
        )

    follow_fraction = np.clip(
        (float(forward_clearance) - FOLLOW_STOP) / (FOLLOW_FULL - FOLLOW_STOP),
        0.0,
        1.0,
    )
    return np.array(
        [0.0, CONFIG["MAXSPEED"] * follow_fraction],
        dtype=np.float32,
    )


def clean_policy_action(state, action):
    """Project learned velocity onto the locally observable movement mode."""
    result = np.asarray(action, dtype=np.float32).copy()
    aisle_offset = float(state[0])
    if abs(aisle_offset) > AISLE_CENTER_STATE_EPS + 1e-6:
        # Lateral merge commands are binary in the expert. Keep their original
        # stop/go threshold separate from the much smaller threshold needed by
        # continuous longitudinal following. Once an agent occupies the aisle,
        # allow a slow lateral command to finish the last few pixels of centering.
        occupies_aisle = abs(aisle_offset) <= AISLE_OCCUPANCY_STATE_EPS + 1e-6
        threshold = (
            POLICY_STOP_THRESHOLD
            if occupies_aisle else LATERAL_POLICY_STOP_THRESHOLD
        )
        aisleward_speed = float(result[0]) * np.sign(aisle_offset)
        if aisleward_speed < threshold:
            result[:] = 0.0
            return result
        # The expert never moves longitudinally before centering. Projecting
        # away that learned blend prevents diagonal merge drift and is based
        # only on the signed aisle offset already supplied to the network.
        result[1] = 0.0
    else:
        # Once centered, longitudinal following may legitimately be slow, but
        # backward commands and lateral regression leakage are never valid.
        if float(result[1]) < POLICY_STOP_THRESHOLD:
            result[:] = 0.0
            return result
        result[0] = 0.0
    return result


def scale_action_for_agent(agent, action):
    """Convert a canonical 5 px/s action to this passenger's preferred speed."""
    preferred_speed = agent.getData("preferred_speed")
    if preferred_speed is None:
        preferred_speed = CONFIG["MAXSPEED"]
    scale = float(preferred_speed) / CONFIG["MAXSPEED"]
    return np.asarray(action, dtype=np.float32) * scale


def _sample_preferred_speed(rng):
    """Sample persistent speed uniformly across the configured range."""
    return rng.uniform(PREFERRED_SPEED_MIN, PREFERRED_SPEED_MAX)


def sample_training_examples(samples_per_regime, rng):
    """Sample the complete observable rule without adding hidden information."""
    state_groups = [[], [], []]
    action_groups = [[], [], []]

    while any(len(group) < samples_per_regime for group in state_groups):
        batch = rng.uniform(
            low=[-1.0, 0.0, 0.0, 0.0],
            high=[1.0, 1.0, 1.0, 1.0],
            size=(max(4096, samples_per_regime), STATE_DIM),
        ).astype(np.float32)
        actions = np.asarray([expert_action(state) for state in batch])
        speeds = np.linalg.norm(actions, axis=1)
        labels = np.zeros(len(actions), dtype=np.int8)
        moving = speeds >= ACTION_REGIME_THRESHOLD
        labels[moving & (np.abs(actions[:, 0]) >= np.abs(actions[:, 1]))] = 1
        labels[moving & (np.abs(actions[:, 0]) < np.abs(actions[:, 1]))] = 2

        for label in range(3):
            needed = samples_per_regime - len(state_groups[label])
            if needed <= 0:
                continue
            selected = np.flatnonzero(labels == label)[:needed]
            state_groups[label].extend(batch[selected])
            action_groups[label].extend(actions[selected])

    states = np.asarray(
        [state for group in state_groups for state in group],
        dtype=np.float32,
    )
    actions = np.asarray(
        [action for group in action_groups for action in group],
        dtype=np.float32,
    )
    order = rng.permutation(len(states))
    return states[order], actions[order]


def sample_boundary_training_examples(samples_per_side, bottom_row_repeats, rng):
    """Densely sample both sides of each observable expert decision boundary."""
    offsets = np.linspace(0.001, 0.02, samples_per_side, dtype=np.float32)
    # The expert moves at equality because each comparison is strictly '<'.
    # Repeat exact values so the MLP does not smooth that semantic edge away.
    offsets[:max(1, samples_per_side // 20)] = 0.0
    states = []

    # Same-side row clearance: keep merge and forward conditions safely clear.
    for direction in (-1.0, 1.0):
        for sign in (-1.0, 1.0):
            for offset in offsets:
                states.append(
                    [
                        direction * rng.uniform(AISLE_CENTER_STATE_EPS + 0.02, 1.0),
                        ROW_SAFE + sign * offset,
                        rng.uniform(MERGE_SAFE + 0.02, 1.0),
                        rng.uniform(FOLLOW_SAFE + 0.02, 1.0),
                    ]
                )

    # Opposing merge clearance: the right side changes action at MERGE_SAFE.
    for sign in (-1.0, 1.0):
        for offset in offsets:
            states.append(
                [
                    -rng.uniform(AISLE_CENTER_STATE_EPS + 0.02, 1.0),
                    rng.uniform(ROW_SAFE + 0.02, 1.0),
                    MERGE_SAFE + sign * offset,
                    rng.uniform(FOLLOW_SAFE + 0.02, 1.0),
                ]
            )

    # Sample both ends of the proportional following interval.
    for boundary in (FOLLOW_STOP, FOLLOW_FULL):
        for sign in (-1.0, 1.0):
            for offset in offsets:
                states.append(
                    [
                        rng.uniform(
                            -AISLE_CENTER_STATE_EPS * 0.5,
                            AISLE_CENTER_STATE_EPS * 0.5,
                        ),
                        rng.uniform(0.0, 1.0),
                        rng.uniform(0.0, 1.0),
                        boundary + sign * offset,
                    ]
                )

    # Repeated exact initial observations make the visually important first
    # bottom-row yield explicit without introducing identity or schedule data.
    bottom_row_agents, _ = build_agents(count=6)
    bottom_states = [extract_state(agent, bottom_row_agents) for agent in bottom_row_agents]
    for _ in range(bottom_row_repeats):
        states.extend(bottom_states)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray([expert_action(state) for state in states], dtype=np.float32)
    order = rng.permutation(len(states))
    return states[order], actions[order]


def extract_action(agent):
    physical = np.array([agent.velocity.x, agent.velocity.y], dtype=np.float32)
    preferred_speed = agent.getData("preferred_speed")
    if preferred_speed is None:
        return physical
    return physical * (CONFIG["MAXSPEED"] / float(preferred_speed))


def filter_agent(agent):
    return agent.position.y < EXIT_Y


def initialize_training_episode(agents, rng):
    """Apply the same bounded initial perturbation used by the expert gate."""
    for agent in agents:
        agent.position += Vector2(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))


def perturb_for_training(agent, rng):
    """Disable legacy per-frame random walks for the local airplane expert."""
    return None


def _to_pygame(pos3):
    x, z = pos3[0], pos3[2]
    return Vector2((x + 100) * 3, (z + 100) * 3)


def build_walls(layout=None):
    layout = parse_layout(layout)
    outer = layout.outer_wall_column
    if isinstance(layout, WidebodyLayout):
        walls_data = [
            (90, False, Vector2(outer, -5)),
            (90, False, Vector2(-outer, -5)),
            # Match the 18 px single-aisle exit width at both aisles.
            (11, True, Vector2(-18.5, 40)),
            (14, True, Vector2(0, 40)),
            (11, True, Vector2(18.5, 40)),
            (2 * outer, True, Vector2(0, -50)),
        ]
    else:
        divider_inner = 3.0
        divider_width = outer - divider_inner
        divider_center = (outer + divider_inner) / 2.0
        walls_data = [
            (90, False, Vector2(outer, -5)),
            (90, False, Vector2(-outer, -5)),
            (divider_width, True, Vector2(divider_center, 40)),
            (divider_width, True, Vector2(-divider_center, 40)),
            (2 * outer, True, Vector2(0, -50)),
        ]
    return [
        Wall(width * 3, vertical, _to_pygame((position.x, 0, position.y)))
        for width, vertical, position in walls_data
    ]


def build_agents(count=None, layout=None, occupancy=None, seed=None):
    layout = parse_layout(layout)
    if count is not None and occupancy is not None:
        raise ValueError("provide count or occupancy, not both")
    if count is None:
        if occupancy is None:
            count = layout.capacity
        else:
            if not 0.0 < occupancy <= 1.0:
                raise ValueError("occupancy must be in (0, 1]")
            count = max(1, round(layout.capacity * occupancy))
    if not 0 <= count <= layout.capacity:
        raise ValueError(f"count must be between 0 and {layout.capacity}")

    agents = []
    geometry_rows = list(range(-47, -47 + layout.rows * 5, 5))
    if isinstance(layout, WidebodyLayout):
        seat_specs = layout.seat_specs
    else:
        seat_specs = [(column, 0, 0) for column in layout.seat_columns]
    seat_slots = [
        (row_index, z, seat_column, aisle_index, aisle_column)
        for row_index, z in enumerate(geometry_rows)
        for seat_column, aisle_index, aisle_column in seat_specs
    ]
    occupancy_rng = random.Random(seed) if seed is not None else random
    speed_rng = random.Random(seed) if seed is not None else random
    if occupancy is not None:
        selected_indices = set(occupancy_rng.sample(range(len(seat_slots)), count))
        seat_slots = [
            slot for index, slot in enumerate(seat_slots) if index in selected_indices
        ]
    else:
        seat_slots = seat_slots[:count]

    aisle_positions = {
        aisle_index: _to_pygame((aisle_column, 0, 0)).x
        for _, aisle_index, aisle_column in seat_specs
    }
    for row_index, z, seat_column, aisle_index, aisle_column in seat_slots:
        position = _to_pygame((seat_column, 0, z))
        preferred_speed = _sample_preferred_speed(speed_rng)
        agent = Agent(
            len(agents),
            position,
            CONFIG["RADIUS"],
            preferred_speed,
            CONFIG["MAXFORCE"],
            CONFIG["HORIZON"],
            CONFIG["K"],
            CONFIG["AVOID"],
            CONFIG["SIDESTEP"],
        )
        agent.setData("start_pos", Vector2(position))
        agent.setData("start_row", row_index)
        agent.setData("start_side", 1 if seat_column > aisle_column else -1)
        agent.setData("start_aisle", aisle_index)
        agent.setData("aisle_x", aisle_positions[aisle_index])
        agent.setData("preferred_speed", preferred_speed)
        agents.append(agent)

    return agents, {"layout": layout, "aisle_positions": aisle_positions}


if __name__ == "__main__":
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    walls = build_walls()
    agents, _ = build_agents()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        states = [extract_state(agent, agents) for agent in agents]
        actions = [
            scale_action_for_agent(agent, expert_action(state))
            for agent, state in zip(agents, states)
        ]
        advance_velocity_agents(agents, actions, walls, TIMESTEP)
        agents = [agent for agent in agents if filter_agent(agent)]

        screen.fill((30, 30, 30))
        for wall in walls:
            pygame.draw.rect(screen, (200, 200, 200), wall.rect)
        for agent in agents:
            pygame.draw.circle(
                screen,
                (0, 255, 0),
                (int(agent.position.x), int(agent.position.y)),
                int(agent.radius),
            )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
