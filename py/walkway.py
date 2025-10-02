import pygame
from pygame.math import Vector2
from agent import Agent
from wall import Wall
from physics import update_agent
import utils
import random

CONFIG = {
    "COUNT": 75,
    "RADIUS": 3,
    "MAXSPEED": 15,
    "MAXFORCE": 90,
    "HORIZON": 15,
    "K": 2,
    "AVOID": 15,
    "SIDESTEP": 15,
}

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
LENGTH = 600

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

agents = []
walls = []
positions = {}
frame = 0
pause = False
timestep = 0.016

walls_data = [
    (180, True, Vector2(300, 345)),
    (180, True, Vector2(300, 330)),
    (180, True, Vector2(300, 270)),
    (180, True, Vector2(300, 255)),
]

for length, vertical, pos in walls_data:
    walls.append(Wall(length, vertical, pos))

for i in range(CONFIG["COUNT"]):
    pos = utils.get_position(150, 210, 150, 450)
    max_speed = random.uniform(15, CONFIG["MAXSPEED"])
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
            if agent.position.y > LENGTH - CONFIG["RADIUS"]:
                agent.position.y = LENGTH - CONFIG["RADIUS"]
            elif agent.position.y < CONFIG["RADIUS"]:
                agent.position.y = CONFIG["RADIUS"]

            if agent.position.y > 300:
                agent.target.y = 337.5
            else:
                agent.target.y = 262.5
            agent.target.x = 202.5

            if 210 <= agent.position.x <= 390:
                if 300 < agent.position.y < 345:
                    agent.max_speed = CONFIG["MAXSPEED"] * 1.5
                elif 255 < agent.position.y < 300:
                    agent.max_speed = CONFIG["MAXSPEED"] * 1.5
                else:
                    agent.max_speed = random.uniform(15, CONFIG["MAXSPEED"])

            near_entry = abs(agent.position.x - 202.5) <= 22.5
            left = 330 < agent.position.y < 345
            right = 255 < agent.position.y < 270
            
            if (near_entry and (left or right)) or (agent.position.x >= 210 and (left or right)) or agent.position.x >= 375:
                agent.target.x = 900

        for agent in agents:
            update_agent(agent, agents, timestep)

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