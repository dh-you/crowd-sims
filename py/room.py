import pygame
from pygame.math import Vector2
from agent import Agent
from wall import Wall
from physics import update_agent
import utils
import random

CONFIG = {
    "COUNT": 150,
    "RADIUS": 3,
    "MAXSPEED": 30,
    "MAXFORCE": 50,
    "HORIZON": 10,
    "K": 3,
    "AVOID": 15,
    "SIDESTEP": 15,
}

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
TIMESTEP = 0.05

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

agents = []
walls = []
positions = {}
frame = 0
pause = False

walls_data = [
    (300, True, Vector2(300, 150)),
    (225, False, Vector2(450, 262.5)),
    (225, False, Vector2(150, 262.5)),
    (120, True, Vector2(390, 375)),
    (120, True, Vector2(210, 375)),
    (75, False, Vector2(330, 412.5)),
    (75, False, Vector2(270, 412.5)),
]

for length, vertical, pos in walls_data:
    walls.append(Wall(length, vertical, pos))

for i in range(CONFIG["COUNT"]):
    pos = utils.get_position(175, 425, 175, 275)
    max_speed = random.uniform(1, CONFIG["MAXSPEED"])
    agents.append(Agent(i, pos, CONFIG["RADIUS"], max_speed, CONFIG["MAXFORCE"],
                        CONFIG["HORIZON"], CONFIG["K"], CONFIG["AVOID"], CONFIG["SIDESTEP"]))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause = not pause

    if not pause:
        for agent in agents:
            if agent.position.y < 375:
                if agent.position.x > 330:
                    agent.target.x = 270
                elif agent.position.x < 270:
                    agent.target.x = 330
                else:
                    agent.target.x = 300
                agent.target.y = 405
            
            if agent.position.y >= 375:
                agent.target.y = 900

        for agent in agents:
            update_agent(agent, agents, TIMESTEP)

        for agent in agents:
            for wall in walls:
                wall.collision_resolve(agent)

        frame += 1
        positions[frame] = {agent.id: (agent.position.x, agent.position.y) for agent in agents}

    screen.fill((30, 30, 30))
    for wall in walls:
        pygame.draw.rect(screen, (200, 200, 200), wall.rect)
    for agent in agents:
        pygame.draw.circle(screen, (0, 255, 0), (int(agent.position.x), int(agent.position.y)), int(agent.radius))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()