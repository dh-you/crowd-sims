import pygame
from pygame.math import Vector2
from agent import Agent
from wall import Wall
import utils
import random
import numpy as np
import torch
import torch.nn as nn

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
K_NEIGHBORS = 5
STATE_DIM = 4 + K_NEIGHBORS * 4
ACTION_DIM = 2
HIDDEN_DIM = 128


class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BCPolicy(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load("bc_policy.pt", map_location=device))
model.eval()

stats = np.load("norm_stats.npy", allow_pickle=True).item()
state_mean = stats["state_mean"]
state_std = stats["state_std"]
action_mean = stats["action_mean"]
action_std = stats["action_std"]

print("Model loaded.")


def get_state(agent, agents, k=K_NEIGHBORS):
    goal = agent.target - agent.position
    own_vel = agent.velocity
    neighbors = sorted(
        [a for a in agents if a.id != agent.id],
        key=lambda n: (n.position - agent.position).length_squared()
    )[:k]

    state = [goal.x, goal.y, own_vel.x, own_vel.y]
    for n in neighbors:
        rel_pos = n.position - agent.position
        rel_vel = n.velocity - agent.velocity
        state += [rel_pos.x, rel_pos.y, rel_vel.x, rel_vel.y]
    while len(state) < 4 + k * 4:
        state.append(0.0)
    return state


def bc_update_agent(agent, agents, timestep):
    # normalize state
    state = get_state(agent, agents)
    state_norm = (np.array(state, dtype=np.float32) - state_mean) / state_std
    state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        action_norm = model(state_t).cpu().numpy()[0]

    # denormalize action
    action = action_norm * action_std + action_mean
    vel = Vector2(float(action[0]), float(action[1]))

    if vel.length() > agent.max_speed:
        vel = vel.normalize() * agent.max_speed

    agent.velocity = vel
    agent.position += agent.velocity * timestep


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("BC Policy Inference")

agents = []
walls = []

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
pause = False

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
            bc_update_agent(agent, agents, TIMESTEP)

        for agent in agents:
            for wall in walls:
                wall.collision_resolve(agent)

    screen.fill((30, 30, 30))
    for wall in walls:
        pygame.draw.rect(screen, (200, 200, 200), wall.rect)
    for agent in agents:
        pygame.draw.circle(screen, (255, 100, 0),  # orange = BC policy
                           (int(agent.position.x), int(agent.position.y)), int(agent.radius))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
