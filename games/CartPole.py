import gymnasium as gym
from games import pygameAI
import torch

class CartPole(pygameAI):
    def __init__(self, **kwargs):
        self.env = gym.make("CartPole-v1", **kwargs)
        self.set_obs_space('continuous', 4)
        self.action_shape = 1
        self.set_action_space('discrete', 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def initiate_pygame(self, game_speed=60):
        x = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset()

        obs = torch.tensor(obs, device=self.device)

        return None, None, obs, False, None
    
    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(int(action))

        obs = torch.tensor(obs, device=self.device)

        return reward, 0, obs, terminated or truncated, info