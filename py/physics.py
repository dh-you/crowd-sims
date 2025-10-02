import math
from pygame.math import Vector2

EPSILON = 1e-6

def time_to_collision(agent, neighbor):
    r = agent.radius + neighbor.radius
    w = neighbor.position - agent.position
    c = w.length_squared() - r*r
    if c < 0:
        return 0
    v = agent.velocity - neighbor.velocity
    a = v.length_squared()
    b = w.dot(v)
    discriminant = b*b - a*c
    if discriminant <= 0 or a == 0:
        return math.inf
    tau = (b - math.sqrt(discriminant)) / a
    return tau if tau >= 0 else math.inf

def apply_force(agent, f, timestep):
    if f.length() > agent.max_force:
        f = f.normalize() * agent.max_force
    agent.velocity += f * timestep
    if agent.velocity.length() > agent.max_speed:
        agent.velocity = agent.velocity.normalize() * agent.max_speed
    agent.position += agent.velocity * timestep

def update_agent(agent, agents, timestep):
    agent.goal = agent.target - agent.position
    fx_goal = agent.k * (agent.goal.x - agent.velocity.x)
    fy_goal = agent.k * (agent.goal.y - agent.velocity.y)

    fx_avoid = fy_avoid = 0
    fx_sidestep = fy_sidestep = 0

    for neighbor in agents:
        if neighbor.id == agent.id:
            continue
        t = time_to_collision(agent, neighbor)
        dir_vec = agent.position - neighbor.position
        if dir_vec.length() > 0:
            dir_vec = dir_vec.normalize()
        else:
            dir_vec = Vector2(1, 0)

        # sidestep directions
        left = Vector2(-dir_vec.y, dir_vec.x)
        right = Vector2(dir_vec.y, -dir_vec.x)
        sidestep = left if left.dot(neighbor.velocity) < right.dot(neighbor.velocity) else right

        if 0 <= t <= agent.horizon:
            w = (agent.horizon - t) / (t + EPSILON)
            fx_avoid += agent.avoid * dir_vec.x * w
            fy_avoid += agent.avoid * dir_vec.y * w
            fx_sidestep += agent.sidestep * sidestep.x * w
            fy_sidestep += agent.sidestep * sidestep.y * w

    fx = fx_goal + fx_avoid + fx_sidestep
    fy = fy_goal + fy_avoid + fy_sidestep
    apply_force(agent, Vector2(fx, fy), timestep)