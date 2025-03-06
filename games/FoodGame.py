import pygame
from games import pygameAI

class FoodGame(pygameAI):
    def __init__(self):
        self.window_width = 1280
        self.window_height = 720

        # scores
        self.score = 0
        self.highscore = 0

        # downsize the screen so models can store states more efficiently
        self.pool_factor = 16

        self.set_obs_space('continuous', 
            (int(1280 / self.pool_factor), 
             int(720 / self.pool_factor), 
             3)
        )

        self.set_action_space('discrete', 4)

    def update_score(self):
        self.highscore = max(self.highscore, self.score)
        self.score = 0

    def reset(self, render=False):
        if render:
            pygame.time.wait(int(30000 / self.game_speed))

        obs = self.get_rgb_array()

        return 0, 0, obs, False, None
    
    def step(self, action=None, mode='human', render=True):
        assert mode in ['human', 'ai']

        if mode == 'human':
            keys = pygame.key.get_pressed()
            action_ = 1 if keys[pygame.K_SPACE] else 0
        elif mode == 'ai':
            action_ = action

    def run(self, game_speed=60):
        self.initiate_pygame()
        self.reset(render=True)

        self.game_speed = game_speed

        while True:
            _, _, _, done, _ = self.step(mode="human", render=True)

            if done:
                break