import pygame
from games.pygameai import pygameAI
from objects import Ship, Zone
from utils import intersects

class FishingGame(pygameAI):
    def __init__(self):
        self.window_width = 1280
        self.window_height = 720

        # zone
        self.zone = Zone()

        # player
        self.player = Ship(self.window_height)
        self.player_pos = pygame.Vector2(self.window_width / 2, self.window_height - self.player.radius)

        # hp
        self.max_hp = 100

        self.set_obs_space('continuous', 
            (int(1280 / self.pool_factor), 
             int(720 / self.pool_factor), 
             3)
        )

        self.set_action_space('discrete', 2)

    def reset(self, render=False):
        if render:
            pygame.time.wait(int(30000 / self.game_speed))

        self.player.reset_pos()
        self.zone.reset_pos()

        self.hp = self.max_hp

        obs = self.get_rgb_array()

        return 0, 0, obs, False, None

    def step(self, action=None, mode='human'):
        assert mode in ['human', 'ai']

        if mode == 'human':
            keys = pygame.key.get_pressed()
            action_ = 1 if keys[pygame.K_SPACE] else 0
        elif mode == 'ai':
            action_ = action

        if action_:
            self.player.update_pos('up')
        else:
            self.player.update_pos('down')

        self.player_pos.y = self.player.pos

        self.score += 1

        if intersects(self.zone, self.player.radius, self.player_pos):
            reward = 1
        else:
            reward = -1

        self.update_pipes()

        return reward, ..., ..., ..., ...

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


        self.screen.fill("purple")

        pygame.draw.circle(self.screen, "red", self.player_pos, self.player.radius)

        self.font.render_to(self.screen, (10, 10), "Score: {}".format(self.score), (255, 255, 255))
        self.font.render_to(self.screen, (10, 40), "Highscore: {}".format(self.highscore), (255, 255, 255))
        pygame.display.flip()

        self.clock.tick(self.game_speed)
        pass

    def run(self, game_speed=60):
        self.initiate_pygame()
        self.reset(render=True)

        self.game_speed = game_speed

        while True:
            _, _, _, done, _= self.step(mode="human", render=True)

            if done:
                break