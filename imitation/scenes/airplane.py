import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from pygame.math import Vector2
from core.agent import Agent
from core.wall import Wall
import numpy as np
import random

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

PHASE_WAITING = 0
PHASE_TO_AISLE = 1
PHASE_EXITING = 2

# ── IL interface ──────────────────────────────────────────────────────────────
STATE_DIM = 5
ACTION_DIM = 2

_AISLE_X = 300.0
_EXIT_Y = 420.0
_FINAL_GOAL = Vector2(300, 600)
_AISLE_THRESHOLD = 15.0 
_HORIZON = 100.0


def extract_state(agent, agents):
    """5-D ego-centric state for the airplane de-boarding task."""
    start = agent.getData("start_pos")

    # 1-2. Target Direction (unit vector; no deadzone)
    to_target = Vector2(agent.target.x - agent.position.x,
                        agent.target.y - agent.position.y)
    length = to_target.length()
    if length > 1e-6:
        dir_x, dir_y = to_target.x / length, to_target.y / length
    else:
        dir_x, dir_y = 0.0, 0.0

    # 3. Global progress: Distance to the FINAL goal in the lobby
    dist_to_exit = agent.position.distance_to(_FINAL_GOAL)

    # 4 & 5. Row-mate sensors
    side_of_aisle = -1 if start.x < _AISLE_X else 1
    agent_seat_dist = abs(start.x - _AISLE_X)
    blocking_row_dist = _HORIZON
    for other in agents:
        if other is agent or getattr(other, "state", None) == "EXITED": 
            continue
        other_start = other.getData("start_pos")
        if other_start is None or abs(other_start.y - start.y) > 5: 
            continue
        if (-1 if other_start.x < _AISLE_X else 1) != side_of_aisle: 
            continue

        if abs(other_start.x - _AISLE_X) < agent_seat_dist:
            other_in_aisle = abs(other.position.x - _AISLE_X) < _AISLE_THRESHOLD
            if not other_in_aisle:
                blocking_row_dist = min(blocking_row_dist, agent.position.distance_to(other.position))
            else:
                # Vertical clearance: only block if they are still "in front" of our exit path
                if (other.position.y - agent.position.y) < 12.0: 
                    blocking_row_dist = min(blocking_row_dist, agent.position.distance_to(other.position))

    # 5. Wide-Angle Aisle Clearance
    # This sensor still checks if the aisle is clear, but we've removed 
    # the "Glass Door" blindness because agents now despawn automatically.
    forward_aisle_clearance = _HORIZON
    _WIDE_AISLE = 45.0 
    for other in agents:
        if other is agent or getattr(other, "state", None) == "EXITED": 
            continue

        if abs(other.position.x - _AISLE_X) < _WIDE_AISLE:
            y_diff = other.position.y - agent.position.y
            # Check if they are in front of us or slightly behind (buffer)
            if y_diff > -15.0:
                d = agent.position.distance_to(other.position)
                if d < forward_aisle_clearance:
                    forward_aisle_clearance = d

    return np.array([
        dir_x, dir_y, dist_to_exit,
        blocking_row_dist, forward_aisle_clearance,
    ], dtype=np.float32)

def extract_action(agent):
    return np.array([agent.velocity.x, agent.velocity.y], dtype=np.float32)

def filter_agent(agent):
    return getattr(agent, "state", None) != "EXITED"

def training_phase(agent):
    """Return auxiliary supervision metadata; this is never a policy input."""
    return {
        "WAITING": PHASE_WAITING,
        "TO_AISLE": PHASE_TO_AISLE,
        "EXITING": PHASE_EXITING,
    }.get(getattr(agent, "state", None), PHASE_WAITING)

def perturb_for_training(agent, rng):
    """Apply covariate-shift noise without teaching motion through seat rows."""
    phase = training_phase(agent)
    if phase == PHASE_WAITING:
        return
    agent.position.x += rng.uniform(-2, 2)
    if phase == PHASE_EXITING:
        agent.position.y += rng.uniform(-2, 2)

def expert_should_move(agent):
    return training_phase(agent) != PHASE_WAITING

def clean_expert_motion(agent):
    """Make expert labels obey the airplane's seat-row geometry."""
    start = agent.getData("start_pos")
    phase = training_phase(agent)
    if phase == PHASE_WAITING:
        agent.position.update(start)
        agent.velocity.update(0, 0)
    elif phase == PHASE_TO_AISLE:
        agent.position.y = start.y
        agent.velocity.y = 0.0

# ── Scene Logic ───────────────────────────────────────────────────────────────

def _to_pygame(pos3):
    x, z = pos3[0], pos3[2]
    return Vector2((x + 100) * 3, (z + 100) * 3)

def build_walls():
    # Side walls and cabin dividers; the lobby remains open.
    walls_data = [
        (90, 4, False, Vector2(14, -5)), 
        (90, 4, False, Vector2(-14, -5)),
        (11, 4, True, Vector2(8.5, 40)), 
        (11, 4, True, Vector2(-8.5, 40)),
        (28, 4, True, Vector2(0, -50))
    ]
    return [Wall(width * 3, vertical, _to_pygame((pos.x, 0, pos.y))) for width, _, vertical, pos in walls_data]

def _shuffle_row(leftRows, rightRows, aisle, rowNum, left):
    row = leftRows[rowNum] if left else rightRows[rowNum]
    nextAgent = row.pop() if row else None
    if nextAgent:
        aisle[rowNum] = nextAgent

def build_agents(count=None):
    if count is None:
        count = CONFIG["COUNT"]

    agents = []
    leftRows, rightRows, orders, aisle = [], [], [], []

    # Full airplane layout: rows from front to back at 5-unit spacing.
    geometry_rows = list(range(-47, 37, 5))
    seat_cols = [12, 8, 4, -12, -8, -4]

    agent_id = 0

    for i, z in enumerate(geometry_rows):
        left_side = []
        right_side = []

        for j in seat_cols:
            if agent_id >= count:
                break

            start_pos = _to_pygame((j, 0, z))
            agent = Agent(
                agent_id,
                start_pos,
                CONFIG["RADIUS"],
                CONFIG["MAXSPEED"],
                CONFIG["MAXFORCE"],
                CONFIG["HORIZON"],
                CONFIG["K"],
                CONFIG["AVOID"],
                CONFIG["SIDESTEP"],
            )

            agent.setData("start_pos", Vector2(start_pos.x, start_pos.y))
            agent.state = "WAITING"
            agents.append(agent)

            # IMPORTANT:
            # j > 0 is right side, j < 0 is left side.
            # Lists are ordered window -> aisle, so pop() releases aisle-closest first.
            if j > 0:
                right_side.append(agent)   # [12, 8, 4], pop gives 4 first
            else:
                left_side.append(agent)    # [-12, -8, -4], pop gives -4 first

            agent_id += 1

        if left_side or right_side:
            leftRows.append(left_side)
            rightRows.append(right_side)

            # True means choose from leftRows, False means choose from rightRows
            row_order = [True] * len(left_side) + [False] * len(right_side)
            random.shuffle(row_order)

            orders.append(row_order)
            aisle.append(None)

    for i in range(len(aisle)):
        if orders[i]:
            _shuffle_row(leftRows, rightRows, aisle, i, orders[i].pop())

    return agents, {
        "leftRows": leftRows,
        "rightRows": rightRows,
        "orders": orders,
        "aisle": aisle,
    }

class SceneController:
    def __init__(self, agents, meta):
        self.leftRows, self.rightRows = meta["leftRows"], meta["rightRows"]
        self.orders, self.aisle = meta["orders"], meta["aisle"]
        self.rowNum = len(self.aisle) - 1

    def update(self, agents):
        if self.rowNum >= 0:
            agentInAisle = self.aisle[self.rowNum]
            if not agentInAisle and self.orders[self.rowNum]:
                _shuffle_row(self.leftRows, self.rightRows, self.aisle, self.rowNum, self.orders[self.rowNum].pop())
            agentInAisle = self.aisle[self.rowNum]
            if agentInAisle:
                if abs(agentInAisle.position.x - _AISLE_X) < 5:
                    # Continue through the door before despawning.
                    agentInAisle.target.y = _FINAL_GOAL.y 
                    agentInAisle.state = "EXITING"
                    self.aisle[self.rowNum] = None
                else:
                    agentInAisle.state = "TO_AISLE"
                    agentInAisle.target.x = _AISLE_X
            elif not self.orders[self.rowNum]:
                self.rowNum -= 1

        for agent in agents:
            if getattr(agent, "state", None) == "EXITING":
                # Vanish instantly at the door line
                if agent.position.y >= _EXIT_Y:
                    agent.state = "EXITED"
                # Keep them locked in the aisle while inside
                elif agent.position.y < _EXIT_Y:
                    agent.position.x = max(min(agent.position.x, 310), 290)       
                                 
if __name__ == "__main__":
    import pygame
    from core.physics import update_agent
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    walls = build_walls(); agents, meta = build_agents(); controller = SceneController(agents, meta)
    while True:
        if any(event.type == pygame.QUIT for event in pygame.event.get()): break
        controller.update(agents)
        for agent in agents: update_agent(agent, agents, TIMESTEP)
        for agent in agents:
            for wall in walls: wall.collision_resolve(agent)
        screen.fill((30, 30, 30))
        for wall in walls: pygame.draw.rect(screen, (200, 200, 200), wall.rect)
        for agent in agents: pygame.draw.circle(screen, (0, 255, 0), (int(agent.position.x), int(agent.position.y)), int(agent.radius))
        pygame.display.flip(); clock.tick(60)
    pygame.quit()
