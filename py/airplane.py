import pygame
from pygame.math import Vector2
from agent import Agent
from wall import Wall
from physics import update_agent
import utils
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

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

agents = []
walls = []
leftRows = []
rightRows = []
orders = []
aisle = []
rowNum = 0
positions = {}
frame = 0

EPSILON = 0.1

def to_pygame(pos3):
    x, z = pos3[0], pos3[2]
    return Vector2((x + 100) * 3, (z + 100) * 3)

wallsData = [
    (100, 4, False, Vector2(14, 0)),
    (110, 4, False, Vector2(-14, 5)),
    (11, 4, True, Vector2(8.5, 40)),
    (11, 4, True, Vector2(-8.5, 40)),
    (28, 4, True, Vector2(0, -50)),
    (28, 4, True, Vector2(0, 60))
]

for width, height, vertical, pos in wallsData:
    walls.append(Wall(width * 3, vertical, to_pygame((pos.x, 0, pos.y))))

floor_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

geometry_rows = list(range(-47, 37, 5))
seat_cols = [12, 8, 4, -12, -8, -4]

for i, z in enumerate(geometry_rows):
    row = []
    for j in seat_cols:
        start_pos = to_pygame((j, 0, z))
        maxSpeed = CONFIG["MAXSPEED"] + random.uniform(-2, 2)
        k = CONFIG["K"] + random.uniform(-1.5, 1.5)
        horizon = CONFIG["HORIZON"] + random.uniform(-5, 5)
        agent = Agent(
            i * len(seat_cols) + j,
            start_pos,
            CONFIG["RADIUS"],
            maxSpeed,
            CONFIG["MAXFORCE"],
            horizon,
            k,
            CONFIG["AVOID"],
            CONFIG["SIDESTEP"]
        )
        row.append(agent)
        agents.append(agent)
    leftRows.append(row[:3])
    rightRows.append(row[3:])
    orders.append([True, True, True, False, False, False])
    random.shuffle(orders[-1])
    aisle.append(None)

rowNum = len(aisle) - 1

def shuffle(rowNum, left):
    row = leftRows[rowNum] if left else rightRows[rowNum]
    for agent in row:
        agent.target.x += -4 if left else 4
    nextAgent = row.pop() if row else None
    if nextAgent:
        aisle[rowNum] = nextAgent

for i in range(len(aisle)):
    shuffle(i, orders[i].pop())

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if rowNum >= 0:
        agentInAisle = aisle[rowNum]
        if not agentInAisle and orders[rowNum]:
            shuffle(rowNum, orders[rowNum].pop())
        agentInAisle = aisle[rowNum]
        if agentInAisle:
            reachedAisle = abs(agentInAisle.position.x - 300) < 5
            if not reachedAisle:
                agentInAisle.target.x = 300
            else:
                agentInAisle.target.y = 460
                agentInAisle.state = "EXITING"
                aisle[rowNum] = None
        elif not orders[rowNum]:
            rowNum -= 1

    for agent in agents:
        if getattr(agent, "state", None) == "EXITING":
            if agent.position.y >= 460:
                agent.target.x = 800
                agent.state = "EXITED"
            elif agent.position.y < 455:
                agent.position.x = max(min(agent.position.x, 315), 285)

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