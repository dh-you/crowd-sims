from pygame.math import Vector2

class Agent:
    def __init__(self, id, position, radius, max_speed, max_force, horizon, k, avoid, sidestep):
        self.id = id
        self.position = position
        self.velocity = Vector2(0, 0)
        self.goal = Vector2(position.x, position.y)
        self.target = Vector2(position.x, position.y)
        self.radius = radius
        self.max_speed = max_speed
        self.max_force = max_force
        self.horizon = horizon
        self.k = k
        self.avoid = avoid
        self.sidestep = sidestep