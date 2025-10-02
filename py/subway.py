import pygame
from pygame.math import Vector2
from agent import Agent
from wall import Wall
from physics import update_agent
import utils
import random

CONFIG = {
    "COUNT": 20,
    "RADIUS": 3,
    "MAXSPEED": 15,
    "MAXFORCE": 50,
    "HORIZON": 7.5,
    "K": 5,
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
assigned = []
animation_done = False

gaps = [-33, -11, 11, 33]

def nearest_door(z):
    return min(gaps, key=lambda g: abs(z - g))

def to_pygame(pos3):
    x, z = pos3[0], pos3[1]
    return Vector2((x + 100) * 3, (z + 100) * 3)

wall_data = [
    (100, False, Vector2(50, 0)),
    (15, True, Vector2(42.5, -50)),
    (15, True, Vector2(42.5, 50)),
    (12, False, Vector2(35, -44)),
    (12, False, Vector2(35, -22)),
    (12, False, Vector2(35, 0)),
    (12, False, Vector2(35, 22)),
    (12, False, Vector2(35, 44)),
]

wall_targets = []
for width, vertical, pos in wall_data:
    start_pos = to_pygame((pos.x, pos.y + 200))
    walls.append(Wall(width * 3, vertical, start_pos))
    wall_targets.append(to_pygame((pos.x, pos.y)))

agents_targets = []

for i in range(CONFIG["COUNT"]):
    pos = utils.get_position(40, 47.5, -39, 39)
    start_pos = to_pygame((pos[0], pos[1] + 200))
    agent = Agent(i, start_pos, CONFIG["RADIUS"], random.uniform(5, CONFIG["MAXSPEED"]),
                  CONFIG["MAXFORCE"], CONFIG["HORIZON"], CONFIG["K"], CONFIG["AVOID"], CONFIG["SIDESTEP"])
    agent.group = 1
    agent.target = start_pos.copy()
    agents.append(agent)
    agents_targets.append(to_pygame(pos))
    assigned.append(False)

for i in range(CONFIG["COUNT"]):
    pos = utils.get_position(0, 23, -39, 39)
    agent = Agent(i + CONFIG["COUNT"], to_pygame(pos), CONFIG["RADIUS"], random.uniform(5, CONFIG["MAXSPEED"]),
                  CONFIG["MAXFORCE"], CONFIG["HORIZON"], CONFIG["K"], CONFIG["AVOID"], CONFIG["SIDESTEP"])
    agent.group = 2
    agent.target = agent.position.copy()
    agents.append(agent)
    assigned.append(False)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not animation_done:
        done = True
        for i, wall in enumerate(walls):
            wall.float_pos = wall.float_pos.lerp(wall_targets[i], 0.1)
            wall.update_center()
            if (wall.float_pos - wall_targets[i]).length() > 1:
                done = False

        for i, agent in enumerate(agents[:CONFIG["COUNT"]]):
            agent.position = agent.position.lerp(agents_targets[i], 0.1)
            if (agent.position - agents_targets[i]).length() > 1:
                done = False
            agent.target = agents_targets[i].copy()

        if done:
            animation_done = True
            assigned = [False] * len(agents)
    else:
        for agent in agents:
            if not assigned[agent.id]:
                if agent.group == 1:
                    door_y = nearest_door((agent.position.y / 3) - 100)
                    agent.target = to_pygame((25, door_y))
                else:
                    door_y = nearest_door((agent.position.y / 3) - 100)
                    agent.target = to_pygame((42, door_y))
                    agent.phase = "door"
                assigned[agent.id] = True
            else:
                if agent.group == 1:
                    if (agent.target - agent.position).length() < 5:
                        agent.target.x -= 50
                else:
                    if agent.phase == "door" and (agent.target - agent.position).length() < 5:
                        agent.target = to_pygame(utils.get_position(40, 47.5, -39, 39))
                        agent.phase = "inside"

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
        color = (0, 255, 0) if agent.group == 1 else (255, 0, 0)
        pygame.draw.circle(screen, color, (int(agent.position.x), int(agent.position.y)), int(agent.radius))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()