import pygame
from games.pygameai import pygameAI
from games.objects import TiledPlayer, TiledFood

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

        self.tile_size = 20
        self.player = TiledPlayer(tile_rows=3, tile_cols=20, tile_size=64)
        self.food = TiledFood(tile_rows=3, tile_cols=20, tile_size=64)
        self.food.spawn(self.player.tile_row, self.player.tile_col)

        self.set_action_space('discrete', 4)

        self.name = "FoodGame"

    def update_score(self):
        self.highscore = max(self.highscore, self.score)
        self.score = 0

    def reset(self, render=False):
        if render:
            pygame.time.wait(int(30000 / self.game_speed))

        self.player.reset_pos()
        self.food.spawn()

        self.time = 300

        self.action = 0

        obs = self.get_rgb_array()

        return 0, 0, obs, False, None
    
    def step(self, action=None, mode='human', render=True):
        assert mode in ['human', 'ai']

        if mode == 'human':
            keys = pygame.key.get_pressed()

            if keys[pygame.K_w]:
                self.action = 0
            elif keys[pygame.K_a]:
                self.action = 1
            elif keys[pygame.K_s]:
                self.action = 2
            elif keys[pygame.K_d]:
                self.action = 3

        elif mode == 'ai':
            self.action = action
        
        reward = -0.1
        done = False

        self.player.move(self.action)
        if self.player.tile_row == self.food.tile_row and self.player.tile_col == self.food.tile_col:
            reward = 1
            self.score += 1
            self.food.spawn()

        if render:
            self.render()
        
        obs = self.get_rgb_array()
        self.time -= 1
        if self.time <= 0:
            done = True
            self.update_score()
            self.reset(render=render)
        
        return reward, 0, obs, done, None


    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill("purple")

        pygame.draw.ellipse(self.screen, (0, 255, 0), self.player)
        pygame.draw.rect(self.screen, (255, 0, 0), self.food)

        self.font.render_to(self.screen, (10, 10), "Score: {}".format(self.score), (255, 255, 255))
        self.font.render_to(self.screen, (10, 40), "Highscore: {}".format(self.highscore), (255, 255, 255))
        self.font.render_to(self.screen, (10, 70), "Time: {}".format(self.time), (255, 255, 255))
        pygame.display.flip()

        self.clock.tick(self.game_speed)

    def run(self, game_speed=15):
        self.initiate_pygame()
        self.reset(render=True)

        self.game_speed = game_speed

        while True:
            _, _, _, done, _ = self.step(mode="human", render=True)

            if done:
                break