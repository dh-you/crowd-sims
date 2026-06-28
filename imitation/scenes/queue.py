import os
import random
import sys

import numpy as np
from pygame.math import Vector2

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.agent import Agent
from core.physics import update_agent


CONFIG = {
    "COUNT": 10,
    "RADIUS": 3,
    "MAXSPEED": 30,
    "MAXFORCE": 50,
    "HORIZON": 10,
    "K": 3,
    "AVOID": 15,
    "SIDESTEP": 15,
}

QUEUE_START_X = 500
QUEUE_Y = 300
SLOT_SPACING = 32

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
EXIT_THRESHOLD_X = SCREEN_WIDTH - 50
TIMESTEP = 0.05

FORWARD_CLEARANCE_HORIZON = 100.0

# State: target offset (2) + clearance to the nearest person ahead (1).
STATE_DIM = 3
ACTION_DIM = 2


def extract_state(agent, agents):
    """Return [target_dx, target_dy, forward_clearance]."""
    target_offset = agent.target - agent.position
    forward_clearance = FORWARD_CLEARANCE_HORIZON

    for other in agents:
        if other is agent or not filter_agent(other):
            continue

        # The queue advances toward positive x, so only agents to the right
        # are ahead. Euclidean distance still captures lateral perturbations.
        if other.position.x > agent.position.x:
            distance = agent.position.distance_to(other.position)
            forward_clearance = min(forward_clearance, distance)

    return np.array(
        [target_offset.x, target_offset.y, forward_clearance],
        dtype=np.float32,
    )


def extract_action(agent):
    """Return the expert's resulting velocity."""
    return np.array([agent.velocity.x, agent.velocity.y], dtype=np.float32)


def filter_agent(agent):
    return getattr(agent, "state", None) != "EXITED"


def build_walls():
    return []


def build_agents(count=None):
    if count is None:
        count = CONFIG["COUNT"]

    agents = []
    for i in range(count):
        position = Vector2(QUEUE_START_X - i * SLOT_SPACING, QUEUE_Y)
        max_speed = random.uniform(15, CONFIG["MAXSPEED"])
        agents.append(
            Agent(
                i,
                position,
                CONFIG["RADIUS"],
                max_speed,
                CONFIG["MAXFORCE"],
                CONFIG["HORIZON"],
                CONFIG["K"],
                CONFIG["AVOID"],
                CONFIG["SIDESTEP"],
            )
        )

    return agents, {}


class SceneController:
    def __init__(self, agents, meta):
        pass

    def update(self, agents):
        # An agent that crossed the exit on the previous physics step leaves
        # before targets are assigned, causing everyone else to advance a slot.
        while agents and agents[0].position.x >= EXIT_THRESHOLD_X - agents[0].radius:
            agents[0].state = "EXITED"
            agents.pop(0)

        for index, agent in enumerate(agents):
            if index == 0:
                agent.target = Vector2(EXIT_THRESHOLD_X, QUEUE_Y)
            else:
                agent.target = Vector2(
                    QUEUE_START_X - index * SLOT_SPACING,
                    QUEUE_Y,
                )


if __name__ == "__main__":
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Queue")

    agents, meta = build_agents()
    controller = SceneController(agents, meta)
    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused

        if not paused:
            controller.update(agents)
            for agent in agents:
                update_agent(agent, agents, TIMESTEP)

        screen.fill((30, 30, 30))
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
