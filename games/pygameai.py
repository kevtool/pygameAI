import pygame
import torch

class pygameAI():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # returning torch tensor
    def get_rgb_array(self):
        # return pygame.surfarray.array3d(self.screen)
        array = pygame.surfarray.array3d(self.screen)

        array = torch.tensor(array, dtype=torch.float32, device=self.device)
        array = array.permute(2, 0, 1).unsqueeze(0)
        avg_pool = torch.nn.AvgPool2d(kernel_size=self.pool_factor, stride=self.pool_factor)
        array = avg_pool(array)
        # array.to('cpu')

        return array


    def set_game_speed(self, game_speed):
        self.game_speed = game_speed

    def set_action_space(self, type, shape):
        assert type in ['discrete', 'continuous']
        self.action_type = type
        self.action_shape = shape

        if type == 'discrete':
            self.action_space = [i for i in range(shape)]
        elif type == 'continuous':
            pass

    def set_obs_space(self, type, shape):
        assert type in ['discrete', 'continuous']
        self.obs_type = type
        self.obs_shape = shape

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass