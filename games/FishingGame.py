import pygame
from games.pygameai import pygameAI
from objects import Ship

class FishingGame(pygameAI):
    def __init__(self):
        self.window_width = 1280
        self.window_height = 720

        # player
        self.player = Ship(self.window_height)
        self.player_pos = pygame.Vector2(self.window_width / 4, self.window_height - self.player.radius)

        self.set_obs_space('continuous', 
            (int(1280 / self.pool_factor), 
             int(720 / self.pool_factor), 
             3)
        )

        self.set_action_space('discrete', 2)

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass