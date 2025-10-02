import pygame
from pygame.math import Vector2
from agent import Agent
from wall import Wall
from physics import update_agent
import utils
import random
from poisson_disc import Bridson_sampling
import numpy as np

CONFIG = {
    "COUNT": 150,
    "RADIUS": 3,
    "MAXSPEED": 30,
    "MAXFORCE": 150,
    "HORIZON": 3,
    "K": 3,
    "AVOID": 15,
    "SIDESTEP": 15,
    "TRANSITIONCAP": 20,
    "wAgent": 30,
    "wPainting": 80,
    "MINCOMFORT": 30,
    "MIDCOMFORT": 60,
    "MAXCOMFORT": 90,
    "BLOCKED_THRESH": 1,
}

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
EPSILON = 0.01
TIMESTEP = 0.05

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

agents = []
walls = []
paintings = []
viewing_points = []
exit_points = []
viewer_counts = {}
in_transition = 0
positions = {}
frame = 0
pause = False

def weighted_score(point, agent, painting):
    return CONFIG["wAgent"] * point.distance_to(agent.position) + CONFIG["wPainting"] * point.distance_to(painting["position"])

def generate_viewing_position(agent, painting):
    if not viewing_points:
        return Vector2(300, 300)
    sorted_points = sorted(viewing_points, key=lambda p: weighted_score(p, agent, painting), reverse=True)
    best_point = sorted_points[-1]
    viewing_points.remove(best_point)
    return best_point.copy()

def generate_exit_position(agent):
    sorted_points = sorted(exit_points, key=lambda p: p.distance_to(agent.position))
    return sorted_points[0].copy() if sorted_points else Vector2(300, 300)

def choose_painting(agent):
    unblocked_paintings = []
    
    for painting in paintings:
        if painting["id"] != agent.data.get("painting"):
            unblocked_paintings.append(painting)
    
    min_viewers = CONFIG["COUNT"]
    best_painting = random.choice(paintings) if paintings else None
    
    for painting in unblocked_paintings:
        if viewer_counts[painting["id"]] < min_viewers:
            min_viewers = viewer_counts[painting["id"]]
            best_painting = painting
    
    return best_painting

walls_data = [
    (300, True, Vector2(300, 150)),
    (300, False, Vector2(450, 300)),
    (300, False, Vector2(150, 300)),
    (300, True, Vector2(300, 450)),
]

for length, vertical, pos in walls_data:
    walls.append(Wall(length, vertical, pos))

paintings_data = [
    (20, False, Vector2(((-50 + EPSILON) + 100) * 3, (-33 + 100) * 3)),
    (20, False, Vector2(((-50 + EPSILON) + 100) * 3, (-11 + 100) * 3)),
    (20, False, Vector2(((-50 + EPSILON) + 100) * 3, (11 + 100) * 3)),
    (20, False, Vector2(((-50 + EPSILON) + 100) * 3, (33 + 100) * 3)),
    (20, False, Vector2(((50 - EPSILON) + 100) * 3, (-33 + 100) * 3)),
    (20, False, Vector2(((50 - EPSILON) + 100) * 3, (-11 + 100) * 3)),
    (20, False, Vector2(((50 - EPSILON) + 100) * 3, (11 + 100) * 3)),
    (20, False, Vector2(((50 - EPSILON) + 100) * 3, (33 + 100) * 3)),
    (20, True, Vector2((-33 + 100) * 3, ((-50 + EPSILON) + 100) * 3)),
    (20, True, Vector2((-11 + 100) * 3, ((-50 + EPSILON) + 100) * 3)),
    (20, True, Vector2((11 + 100) * 3, ((-50 + EPSILON) + 100) * 3)),
    (20, True, Vector2((33 + 100) * 3, ((-50 + EPSILON) + 100) * 3)),
    (20, True, Vector2((-33 + 100) * 3, ((50 - EPSILON) + 100) * 3)),
    (20, True, Vector2((-11 + 100) * 3, ((50 - EPSILON) + 100) * 3)),
    (20, True, Vector2((11 + 100) * 3, ((50 - EPSILON) + 100) * 3)),
    (20, True, Vector2((33 + 100) * 3, ((50 - EPSILON) + 100) * 3)),
]

painting_id = 0
for size, vertical, pos in paintings_data:
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    painting = {
        "id": painting_id,
        "size": size,
        "vertical": vertical,
        "position": pos,
        "color": color,
    }
    paintings.append(painting)
    viewer_counts[painting_id] = 0
    painting_id += 1

points = Bridson_sampling(np.array([600, 600]), radius=2 * CONFIG["RADIUS"], k=30)
all_points = [Vector2(x, y) for x, y in points]

all_points = [p for p in all_points if 160 < p.x < 440 and 160 < p.y < 440]

for point in all_points:
    too_close = any(painting["position"].distance_to(point) < CONFIG["MINCOMFORT"] for painting in paintings)
    has_nearby = any(painting["position"].distance_to(point) < CONFIG["MIDCOMFORT"] for painting in paintings)
    
    if not too_close and has_nearby:
        viewing_points.append(point)

for point in all_points:
    too_close = any(painting["position"].distance_to(point) < CONFIG["MIDCOMFORT"] for painting in paintings)
    has_far = any(painting["position"].distance_to(point) > CONFIG["MAXCOMFORT"] for painting in paintings)
    
    if not too_close and has_far:
        exit_points.append(point)

for i in range(CONFIG["COUNT"]):
    max_speed = random.uniform(15, CONFIG["MAXSPEED"])
    painting = random.choice(paintings)
    
    start_pos = utils.get_position(175, 425, 175, 425)
    
    temp_agent = Agent(i, start_pos, CONFIG["RADIUS"], max_speed, CONFIG["MAXFORCE"],
                      CONFIG["HORIZON"], CONFIG["K"], CONFIG["AVOID"], CONFIG["SIDESTEP"])
    
    pos = generate_viewing_position(temp_agent, painting)
    
    agent = Agent(i, pos + Vector2(EPSILON, EPSILON), CONFIG["RADIUS"], max_speed, CONFIG["MAXFORCE"],
                  CONFIG["HORIZON"], CONFIG["K"], CONFIG["AVOID"], CONFIG["SIDESTEP"])
    
    agent.target = pos.copy()
    agent.data["viewing_position"] = pos
    agent.data["painting"] = painting["id"]
    agent.data["state"] = "VIEWING"
    agent.data["timer"] = random.uniform(0, 10)
    viewer_counts[painting["id"]] += 1
    
    agents.append(agent)

prev_time = pygame.time.get_ticks()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause = not pause

    if not pause:
        current_time = pygame.time.get_ticks()
        delta = (current_time - prev_time) / 1000.0
        prev_time = current_time

        for agent in agents:
            state = agent.data.get("state")
            
            if state == "VIEWING":
                agent.data["timer"] -= delta * TIMESTEP / 0.005
                
                if agent.data["timer"] <= 0 and in_transition < CONFIG["TRANSITIONCAP"]:
                    in_transition += 1
                    agent.target = generate_exit_position(agent)
                    agent.data["state"] = "EXITING"
                    viewer_counts[agent.data["painting"]] -= 1
                elif agent.data["timer"] <= 0:
                    agent.data["timer"] = random.uniform(0, 10)
            
            elif state == "EXITING":
                if agent.position.distance_to(agent.target) < 15:
                    painting = choose_painting(agent)
                    viewing_points.append(agent.data["viewing_position"])
                    agent.data["viewing_position"] = generate_viewing_position(agent, painting)
                    agent.target = agent.data["viewing_position"].copy()
                    agent.data["painting"] = painting["id"]
                    agent.data["state"] = "WALKING"
                    viewer_counts[agent.data["painting"]] += 1
            
            elif state == "WALKING":
                if agent.position.distance_to(agent.data["viewing_position"]) < 15:
                    in_transition -= 1
                    agent.data["timer"] = random.uniform(0, 10)
                    agent.data["state"] = "VIEWING"

        for agent in agents:
            update_agent(agent, agents, TIMESTEP)

        for agent in agents:
            for wall in walls:
                wall.collision_resolve(agent)

        frame += 1
        positions[frame] = {agent.id: (agent.position.x, agent.position.y) for agent in agents}

    screen.fill((30, 30, 30))
    
    for wall in walls:
        pygame.draw.rect(screen, (100, 100, 100), wall.rect)
    
    for painting in paintings:
        size = int(painting["size"] * 3)
        pos = (int(painting["position"].x), int(painting["position"].y))
        if painting["vertical"]:
            pygame.draw.rect(screen, painting["color"], (pos[0] - size//2, pos[1] - 2, size, 4))
        else:
            pygame.draw.rect(screen, painting["color"], (pos[0] - 2, pos[1] - size//2, 4, size))
    
    for agent in agents:
        pygame.draw.circle(screen, (0, 255, 0), (int(agent.position.x), int(agent.position.y)), int(agent.radius))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()