import pygame
from pygame.math import Vector2
from agent import Agent
from physics import update_agent
import utils
import random
import numpy as np
from poisson_disc import Bridson_sampling

CONFIG = {
    "COUNT": 50,
    "RADIUS": 3,
    "MAXSPEED": 22.5,
    "MAXFORCE": 150,
    "HORIZON": 30,
    "K": 6,
    "AVOID": 15,
    "SIDESTEP": 15,
    "MINCOMFORT": 30,
    "MAXCOMFORT": 75,
    "wAgent": 30,
    "wPerformer": 80,
}

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
LENGTH = 300
EPSILON = 0.01
TIMESTEP = 0.05

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

agents = []
points = []
viewer_counts = {}
positions = {}
frame = 0
selected = None
performer = Vector2(300, 435)

def weighted_score(point, agent, performer):
    return CONFIG["wAgent"] * point.distance_to(agent.position) + CONFIG["wPerformer"] * point.distance_to(performer)

def generate_viewing_position(agent):
    if not points:
        return Vector2(300, 300)
    sorted_points = sorted(points, key=lambda p: weighted_score(p, agent, performer), reverse=True)
    best_point = sorted_points[-1]
    points.remove(best_point)
    return best_point.copy()

def init():
    global points, agents
    for i in range(CONFIG["COUNT"]):
        pos = utils.get_position(105, 495, 255, 345)
        max_speed = random.uniform(15, CONFIG["MAXSPEED"])
        target_x = 150 if random.random() < 0.5 else 450
        agent = Agent(
            i, pos, CONFIG["RADIUS"], max_speed, CONFIG["MAXFORCE"],
            CONFIG["HORIZON"], CONFIG["K"], CONFIG["AVOID"], CONFIG["SIDESTEP"]
        )
        agent.target = Vector2(target_x, pos.y)
        agent.data["state"] = "WALKING"
        agents.append(agent)

    points_data = Bridson_sampling(np.array([600, 600]), radius=2 * CONFIG["RADIUS"], k=20)
    points = [Vector2(x, y) for x, y in points_data]
    points = [p for p in points if CONFIG["MINCOMFORT"] < p.distance_to(performer) < CONFIG["MAXCOMFORT"]]

init()

running = True
prev_time = pygame.time.get_ticks()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = Vector2(pygame.mouse.get_pos())
            selected = None
            for agent in agents:
                if agent.position.distance_to(mouse_pos) < CONFIG["RADIUS"]:
                    selected = agent.id
                    break

    current_time = pygame.time.get_ticks()
    delta = (current_time - prev_time) / 1000.0
    prev_time = current_time

    for agent in agents:
        if agent.position.x < 150:
            agent.position.x = 450
            agent.position.y = 600 - agent.position.y
            agent.target.y = 600 - agent.target.y
        elif agent.position.x > 450:
            agent.position.x = 150
            agent.position.y = 600 - agent.position.y
            agent.target.y = 600 - agent.target.y

        if selected is not None and agent.id == selected and agent.data["state"] == "WALKING":
            agent.data["state"] = "VIEWING"
            agent.target = generate_viewing_position(agent)

        if agent.data["state"] == "VIEWING" and agent.position.y > 360:
            agent.horizon = 3

        update_agent(agent, agents, TIMESTEP)

    frame += 1
    positions[frame] = {agent.id: (agent.position.x, agent.position.y) for agent in agents}

    screen.fill((30, 30, 30))
    pygame.draw.rect(screen, (34, 34, 34), (150, 255, 300, 90))

    pygame.draw.circle(screen, (255, 0, 0), (int(performer.x), int(performer.y)), 5)

    for agent in agents:
        color = (0, 0, 255) if agent.data["state"] == "VIEWING" else (0, 255, 0)
        pygame.draw.circle(screen, color, (int(agent.position.x), int(agent.position.y)), int(agent.radius))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()