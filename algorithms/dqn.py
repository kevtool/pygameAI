import torch
from algorithms.algorithm import Algorithm

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQN(Algorithm):
    def __init__(self, env):
        super().__init__(env)

        policy_network = QNetwork()
        target_network = QNetwork()

    def train(self, num_episodes):
        for epsiode in range(num_episodes):
            _, _, obs, done, _ = self.env.reset()

            while not done:
                action = ...
                reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)