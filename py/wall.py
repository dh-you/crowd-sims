import pygame
from pygame.math import Vector2

class Wall:
    def __init__(self, length, vertical, position, thickness=4):
        if vertical:
            rect_w, rect_h = length, thickness
        else:
            rect_w, rect_h = thickness, length
        self.rect = pygame.Rect(position.x - rect_w/2,
                                position.y - rect_h/2,
                                rect_w, rect_h)

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