import pygame
import pygame.freetype
from pygameai import pygameAI
from objects import Ship, Pipe
from algorithms.genetic import Algorithm

class Game(pygameAI):
    def __init__(self, use_values_for_obs=False):
        self.window_width = 1280
        self.window_height = 720

        # player
        self.player = Ship(self.window_height)
        self.player_pos = pygame.Vector2(self.window_width / 4, self.window_height - self.player.radius)

        # pipes
        self.pipetick = 0
        self.pipes = []

        # scores
        self.score = 0
        self.highscore = 0

        # tracking direction change
        self.prev_action = None
        self.changes = 0

        # obs / action space
        self.use_values_for_obs = use_values_for_obs

        if use_values_for_obs:
            self.set_obs_space('continuous', 4)
        else:
            self.set_obs_space('continuous', (1280, 720, 3))

        self.set_action_space('discrete', 2)

    def initiate_pygame(self, game_speed=60):
        pygame.init()

        # game speed
        self.game_speed = game_speed

        # screen, clock
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.running = True

        # font
        self.font = pygame.freetype.SysFont("Arial", 24)

    def add_pipe(self):
        pipe = Pipe(self.window_width, self.window_height)
        self.pipes.append(pipe)

    def update_pipes(self):
        for pipe in self.pipes:
            pipe.update_pos()

        for i, pipe in enumerate(self.pipes):
            if pipe.pos < -pipe.width:
                del self.pipes[i]

    def draw_pipes(self):
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, "black", pipe.toprect)
            pygame.draw.rect(self.screen, "black", pipe.botrect)

    def intersects(self, rect, r, center):
        circle_distance_x = abs(center[0]-rect.centerx)
        circle_distance_y = abs(center[1]-rect.centery)
        if circle_distance_x > rect.w/2.0+r or circle_distance_y > rect.h/2.0+r:
            return False
        if circle_distance_x <= rect.w/2.0 or circle_distance_y <= rect.h/2.0:
            return True
        corner_x = circle_distance_x-rect.w/2.0
        corner_y = circle_distance_y-rect.h/2.0
        corner_distance_sq = corner_x**2.0 +corner_y**2.0
        return corner_distance_sq <= r**2.0

    def get_nearest_pipe_info(self):
        if len(self.pipes) <= 0:
            return 0, self.window_height, self.window_width

        for pipe in self.pipes:
            if pipe.pos + pipe.width > self.player_pos.x:
                nearest_pipe_topend = pipe.topend
                nearest_pipe_bottomend = pipe.bottomend
                nearest_pipe_x = pipe.pos
                break

        return nearest_pipe_topend, nearest_pipe_bottomend, nearest_pipe_x

    def update_score(self):
        self.highscore = max(self.highscore, self.score)
        self.score = 0

    def get_rgb_array(self):
        # return pygame.surfarray.array3d(self.screen)
        array = pygame.surfarray.array3d(self.screen)
        return array

    def reset(self, render=False):
        if render:
            pygame.time.wait(int(30000 / self.game_speed))

        self.prev_action = None
        self.changes = 0
        self.pipes.clear()
        self.player.reset_pos()
        self.pipetick = 0

        if self.use_values_for_obs == True:
            nearest_pipe_topend, nearest_pipe_bottomend, nearest_pipe_x = self.get_nearest_pipe_info()
            obs = self.normalize_values(self.player_pos.y, nearest_pipe_topend, nearest_pipe_bottomend, nearest_pipe_x)
        else:
            obs = self.get_rgb_array()

        return 0, 0, obs, False, None

    def check_hit(self, player_pos):
        for i, pipe in enumerate(self.pipes):
            if self.intersects(pipe.toprect, self.player.radius, player_pos) or self.intersects(pipe.botrect, self.player.radius, player_pos):
                return True
        
        return False

    def normalize_values(self, *params):
        return [p / self.window_width for p in params]
    

    # the step function of the game needs to take in action as an argument
    # if the mode is human, we can then get action from pygame keys and ignore the argument
    def step(self, action=None, mode='human', brain=None, render=True):
        assert mode in ['human', 'ai']

        if mode == 'human':
            keys = pygame.key.get_pressed()
            action_ = 1 if keys[pygame.K_SPACE] else 0
        elif mode == 'ai':
            # action = (brain.forward(self.normalize_values(self.player_pos.y, nearest_pipe_topend, nearest_pipe_bottomend, nearest_pipe_x)) > 0)
            action_ = action

        assert action_ in self.action_space
        
        if action_:
            self.player.update_pos('up')
        else:
            self.player.update_pos('down')

        change = 0

        if self.prev_action != None:
            if self.prev_action != action_:
                change = 1
                self.changes += 1
            
        self.prev_action = action_
        
        self.player_pos.y = self.player.pos
        
        self.pipetick += 1
        if self.pipetick % 150 == 0:
            self.add_pipe()
            self.pipetick = 0

        self.score += 1

        self.update_pipes()

        if render:
            self.render()

        # new observation
        if self.use_values_for_obs == True:
            nearest_pipe_topend, nearest_pipe_bottomend, nearest_pipe_x = self.get_nearest_pipe_info()
            obs = self.normalize_values(self.player_pos.y, nearest_pipe_topend, nearest_pipe_bottomend, nearest_pipe_x)
        else:
            obs = self.get_rgb_array()

        done = False
        if self.check_hit(self.player_pos) or self.score > 100000:
            done = True
            self.update_score()
            self.reset(render=render)
        
        return 1, change, obs, done, None


    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


        self.screen.fill("purple")

        pygame.draw.circle(self.screen, "red", self.player_pos, self.player.radius)
        self.draw_pipes()
        self.font.render_to(self.screen, (10, 10), "Score: {}".format(self.score), (255, 255, 255))
        self.font.render_to(self.screen, (10, 40), "Highscore: {}".format(self.highscore), (255, 255, 255))
        pygame.display.flip()

        self.clock.tick(self.game_speed)
    
    def run(self, game_speed=60):
        self.initiate_pygame()
        self.reset(render=True)

        self.game_speed = game_speed

        while True:
            _, _, _, done, _= self.step(mode="human", render=True)

            if done:
                break