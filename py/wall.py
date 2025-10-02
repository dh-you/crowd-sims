import pygame
from pygame.math import Vector2

class Wall:
    def __init__(self, length, vertical, position, thickness=4):
        if vertical:
            rect_w, rect_h = length, thickness
        else:
            rect_w, rect_h = thickness, length

        # float position (center of wall)
        self.float_pos = Vector2(position)

        # rect for drawing + collisions
        self.rect = pygame.Rect(0, 0, rect_w, rect_h)
        self.rect.center = (int(position.x), int(position.y))

    def update_center(self):
        """Sync float_pos → rect center (for drawing & collisions)."""
        self.rect.center = (int(self.float_pos.x), int(self.float_pos.y))

    def collision_resolve(self, agent):
        closest_x = max(self.rect.left, min(agent.position.x, self.rect.right))
        closest_y = max(self.rect.top, min(agent.position.y, self.rect.bottom))
        closest = Vector2(closest_x, closest_y)
        push = agent.position - closest
        dist = push.length()
        if dist < agent.radius and dist > 0:
            penetration = agent.radius - dist
            agent.position += push.normalize() * penetration
        elif dist == 0:
            agent.position += Vector2(agent.radius, 0)
