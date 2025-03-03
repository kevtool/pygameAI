import random
import pygame

class Ship():
    def __init__(self, window_height):
        self.window_height = window_height
        self.radius = 30
        self.pos = self.window_height - self.radius
        self.velocity = 0
        self.momentum = 75
        self.max_velocity = 1100

    def update_pos(self, dir):
        if dir == 'up':
            self.velocity -= self.momentum
        elif dir == 'down':
            self.velocity += self.momentum
        else:
            raise Exception("Error: Invalid direction")
        
        self.check_vel_bounds()
        self.pos += self.velocity * 0.016
        self.check_pos_bounds()
    
    def check_vel_bounds(self):
        if self.velocity < -self.max_velocity:
            self.velocity = -self.max_velocity
        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity

    def check_pos_bounds(self):
        if self.pos <= self.radius:
            self.pos = self.radius
            self.velocity = 0
        elif self.pos >= self.window_height - self.radius:
            self.pos = self.window_height - self.radius
            self.velocity = 0

    def reset_pos(self):
        self.pos = self.window_height - self.radius

class Pipe():
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        self.pos = window_width
        self.width = window_width * 0.1
        self.gap = 350
        self.topend = random.randint(40, window_height - (self.gap + 40))
        self.bottomend = self.topend + self.gap
        self.toprect = pygame.Rect(self.pos, 0, self.width, self.topend)
        self.botrect = pygame.Rect(self.pos, self.bottomend, self.width, self.window_height-self.bottomend)

    def update_pos(self):
        self.pos -= 5
        self.toprect = pygame.Rect(self.pos, 0, self.width, self.topend)
        self.botrect = pygame.Rect(self.pos, self.bottomend, self.width, self.window_height-self.bottomend)
    
    def __del__(self):
        return
    
class MovingZone(pygame.Rect):
    def __init__(self, window_width, window_height, width, height, direction='up'):
        self.window_height = window_height

        left = window_width / 2 - width / 2
        top = window_height - height
        super().__init__(left, top, width, height)

        self.init_left = left
        self.init_top = top

        self.init_direction = direction
        self.direction = direction

    # epsilon: chance of direction change
    def update_pos(self, epsilon=0.0):
        if self.top <= 0:
            self.direction = 'down'
        elif self.bottom >= self.window_height:
            self.direction = 'up'

        elif random.random() < epsilon:
            self.direction = 'up' if self.direction == 'down' else 'down'

        if self.direction == 'up':
            self.top -= 4
        else:
            self.top += 4

    def reset_pos(self):
        self.left = self.init_left
        self.top = self.init_top
        self.direction = self.init_direction