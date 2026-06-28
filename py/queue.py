import random

import pygame
from pygame.math import Vector2

from agent import Agent
from physics import update_agent


CONFIG = {
    "RADIUS": 3,
    "MAXSPEED": 30,
    "MAXFORCE": 50,
    "HORIZON": 10,
    "K": 3,
    "AVOID": 15,
    "SIDESTEP": 15,
}

QUEUE_LENGTH = 10
QUEUE_START_X = 500
QUEUE_Y = 300
SLOT_SPACING = 32

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
EXIT_THRESHOLD_X = SCREEN_WIDTH - 50
TIMESTEP = 0.05


def build_agents(count=QUEUE_LENGTH):
    agents = []
    for agent_id in range(count):
        position = Vector2(
            QUEUE_START_X - agent_id * SLOT_SPACING,
            QUEUE_Y,
        )
        max_speed = random.uniform(15, CONFIG["MAXSPEED"])
        agents.append(
            Agent(
                agent_id,
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
    return agents


def update_queue(agents):
    for index, agent in enumerate(agents):
        target_x = (
            EXIT_THRESHOLD_X
            if index == 0
            else QUEUE_START_X - index * SLOT_SPACING
        )
        agent.target = Vector2(target_x, QUEUE_Y)

    for agent in agents:
        update_agent(agent, agents, TIMESTEP)

    if agents and agents[0].position.x >= EXIT_THRESHOLD_X:
        agents.pop(0)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Queue")

    agents = build_agents()
    paused = False
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused

        if not paused and agents:
            update_queue(agents)

        screen.fill((30, 30, 30))
        for agent in agents:
            pygame.draw.circle(
                screen,
                (0, 255, 0),
                (int(agent.position.x), int(agent.position.y)),
                int(agent.radius),
            )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
