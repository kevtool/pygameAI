# ChatGPT assisted

import torch
import random
import numpy as np
from math import prod
from collections import deque
from utils import argmax
from algorithms.algorithm import Algorithm

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

class DQN(Algorithm):
    def __init__(self, env):
        super().__init__(env)

        self.inputs = prod(self.env.obs_shape)
        self.outputs = self.env.action_shape

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.policy_network = QNetwork(self.inputs, self.outputs).to(self.device)
        self.target_network = QNetwork(self.inputs, self.outputs).to(self.device)

    def train(self, num_episodes):
        epsilon = 0
        
        buffer_size = 12000
        replay_buffer = ReplayBuffer(buffer_size)

        self.env.initiate_pygame()

        for ep in range(num_episodes):
            ep_reward = 0
            _, _, obs, done, _ = self.env.reset(render=True)

            while not done:
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        state = np.array(obs, dtype=torch.float32)
                        state_tensor = torch.flatten(torch.tensor(obs, dtype=torch.float32, device='cuda'))

                        action = argmax(self.policy_network(state_tensor).to('cpu'))

                reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)
                next_state = np.array(obs, dtype=torch.float32)
                ep_reward += reward

                replay_buffer.push(state, action, reward, next_state, done)