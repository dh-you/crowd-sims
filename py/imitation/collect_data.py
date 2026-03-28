import pygame
from pygame.math import Vector2
from agent import Agent
from wall import Wall
from physics import update_agent
import utils
import random
import json

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
MAX_FRAMES = 500  # collect 500 frames of data
K_NEIGHBORS = 5  # how many nearest neighbors to include in state

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

agents = []
walls = []
frame = 0
dataset = []  # list of (state, action) pairs

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

def get_state(agent, agents, k=K_NEIGHBORS):
    # state vector, [own_goal_x, own_goal_y, own_vel_x, own_vel_y, 
    # neighbor_rel_pos_x, neighbor_rel_pos_y, neighbor_rel_vel_x, neighbor_rel_vel_y, ...]
    # 4 + 4*k dimensions total (zero-pad if fewer than k neighbors)

    goal = agent.target - agent.position
    own_vel = agent.velocity

    # find k nearest neighbors 
    neighbors = sorted(
        [a for a in agents if a.id != agent.id],
        key=lambda n: (n.position - agent.position).length_squared()
    )[:k]

    state = [goal.x, goal.y, own_vel.x, own_vel.y]

    for n in neighbors:
        rel_pos = n.position - agent.position
        rel_vel = n.velocity - agent.velocity
        state += [rel_pos.x, rel_pos.y, rel_vel.x, rel_vel.y]

    # zero-pad if fewer than k neighbors
    while len(state) < 4 + k * 4:
        state.append(0.0)

    return state

running = True
print("Collecting data...")

while running and frame < MAX_FRAMES:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # collect state before update
    states = {agent.id: get_state(agent, agents) for agent in agents}

    # update agents
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

    # collect action after update: resulting velocity of agents
    for agent in agents:
        action = [agent.velocity.x, agent.velocity.y]
        dataset.append({
            "state": states[agent.id],
            "action": action
        })

    screen.fill((30, 30, 30))
    for wall in walls:
        pygame.draw.rect(screen, (200, 200, 200), wall.rect)
    for agent in agents:
        pygame.draw.circle(screen, (0, 255, 0),
                           (int(agent.position.x), int(agent.position.y)), int(agent.radius))

    pygame.display.flip()
    clock.tick(60)
    frame += 1

    if frame % 50 == 0:
        print(f"Frame {frame}/{MAX_FRAMES}, samples collected: {len(dataset)}")

pygame.quit()

# save dataset
with open("bc_data.json", "w") as f:
    json.dump(dataset, f)

print(f"Done. Saved {len(dataset)} state-action pairs to bc_data.json")
