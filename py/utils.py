import math
import random
from pygame.math import Vector2

EPSILON = 0.005

def get_velocity(max_speed):
    theta = random.random() * 2 * math.pi
    speed = random.random() * max_speed
    return Vector2(speed * math.cos(theta), speed * math.sin(theta))

def get_position(min_x, max_x, min_y, max_y):
    x = random.random() * (max_x - min_x) + min_x
    y = random.random() * (max_y - min_y) + min_y
    return Vector2(x, y)