import os
import sys

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

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
TIMESTEP = 0.05
TRAINING_EPISODE_CAP = 12000

STATE_DIM = 4
ACTION_DIM = 2
POLICY_HIDDEN_DIMS = (32, 32)
POLICY_STOP_THRESHOLD = 2.5
STATE_IS_NORMALIZED = True
BALANCE_ACTION_REGIMES = True
SYNTHETIC_SAMPLES_PER_REGIME = 30000

AISLE_X = 300.0
EXIT_Y = 420.0
MAX_SEAT_DISTANCE = 36.0
SENSOR_RANGE = 100.0
AISLE_HALF_WIDTH = 6.0
FORWARD_CORRIDOR_HALF_WIDTH = 15.0
FORWARD_MIN_Y = 6.0

AISLE_EPS = AISLE_HALF_WIDTH / MAX_SEAT_DISTANCE
ROW_SAFE = 15.0 / SENSOR_RANGE
FOLLOW_SAFE = 18.0 / SENSOR_RANGE
MERGE_SAFE = 30.0 / SENSOR_RANGE


def extract_state(agent, agents):
    """Return four normalized local observations for autonomous deboarding."""
    aisle_delta = AISLE_X - agent.position.x
    aisle_offset = np.clip(
        aisle_delta / MAX_SEAT_DISTANCE,
        -1.0,
        1.0,
    )

    start_row = agent.getData("start_row")
    start_side = agent.getData("start_side")
    row_clearance = SENSOR_RANGE
    opposing_clearance = SENSOR_RANGE
    forward_clearance = SENSOR_RANGE

    direction_to_aisle = np.sign(aisle_delta)
    distance_to_aisle = abs(aisle_delta)

    for other in agents:
        if other is agent or not filter_agent(other):
            continue

        other_row = other.getData("start_row")
        other_side = other.getData("start_side")
        distance = agent.position.distance_to(other.position)

        if other_row == start_row:
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
            other_row != start_row
            and y_diff > FORWARD_MIN_Y
            and abs(other.position.x - AISLE_X)
            <= FORWARD_CORRIDOR_HALF_WIDTH
        ):
            forward_clearance = min(forward_clearance, distance)

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

    if abs(float(aisle_offset)) > AISLE_EPS + 1e-6:
        blocked = (
            row_clearance < ROW_SAFE
            or forward_clearance < FOLLOW_SAFE
            or (aisle_offset < 0 and opposing_clearance < MERGE_SAFE)
        )
        if blocked:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        return np.array(
            [np.sign(aisle_offset) * CONFIG["MAXSPEED"], 0.0],
            dtype=np.float32,
        )

    if forward_clearance < FOLLOW_SAFE:
        return np.zeros(ACTION_DIM, dtype=np.float32)
    return np.array([0.0, CONFIG["MAXSPEED"]], dtype=np.float32)


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
        moving = speeds >= 0.5
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


def extract_action(agent):
    return np.array([agent.velocity.x, agent.velocity.y], dtype=np.float32)


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


def build_walls():
    walls_data = [
        (90, False, Vector2(14, -5)),
        (90, False, Vector2(-14, -5)),
        (11, True, Vector2(8.5, 40)),
        (11, True, Vector2(-8.5, 40)),
        (28, True, Vector2(0, -50)),
    ]
    return [
        Wall(width * 3, vertical, _to_pygame((position.x, 0, position.y)))
        for width, vertical, position in walls_data
    ]


def build_agents(count=None):
    if count is None:
        count = CONFIG["COUNT"]
    if not 0 <= count <= CONFIG["COUNT"]:
        raise ValueError(f"count must be between 0 and {CONFIG['COUNT']}")

    agents = []
    geometry_rows = list(range(-47, 37, 5))
    seat_columns = [12, 8, 4, -12, -8, -4]

    for row_index, z in enumerate(geometry_rows):
        for seat_column in seat_columns:
            if len(agents) >= count:
                return agents, {}

            position = _to_pygame((seat_column, 0, z))
            agent = Agent(
                len(agents),
                position,
                CONFIG["RADIUS"],
                CONFIG["MAXSPEED"],
                CONFIG["MAXFORCE"],
                CONFIG["HORIZON"],
                CONFIG["K"],
                CONFIG["AVOID"],
                CONFIG["SIDESTEP"],
            )
            agent.setData("start_pos", Vector2(position))
            agent.setData("start_row", row_index)
            agent.setData("start_side", 1 if seat_column > 0 else -1)
            agents.append(agent)

    return agents, {}


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
        actions = [expert_action(state) for state in states]
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
