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
        # print(x.shape)
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

        gamma = 0.99
        lr = 1e-3

        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.
        target_update_freq = 10
        
        buffer_size = 12000
        batch_size = 2
        replay_buffer = ReplayBuffer(buffer_size)

        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.env.initiate_pygame()

        for ep in range(num_episodes):
            ep_reward = 0
            _, _, obs, done, _ = self.env.reset(render=True)

            # state = torch.tensor(obs, dtype=torch.float32, device='cuda')
            # state = state.permute(2, 0, 1).unsqueeze(0)
            # print(state.shape)
            # avg_pool = torch.nn.AvgPool2d(kernel_size=16, stride=16)
            # output = avg_pool(state)
            # print(output.shape)
            # quit()

            while not done:
                if random.random() < epsilon:
                    action = random.choice(self.env.action_space)
                else:
                    with torch.no_grad():
                        state_tensor = torch.flatten(torch.tensor(obs, dtype=torch.float32, device='cuda'))

                        action = argmax(self.policy_network(state_tensor).to('cpu'))

                reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)
                next_state = np.array(obs, dtype=float)
                ep_reward += reward

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                if len(replay_buffer) >= batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                    states = torch.tensor(states, dtype=torch.float32).to(self.device)
                    actions = torch.tensor(actions, dtype=torch.long).to(self.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

                    # flatten to fit the network
                    states = torch.flatten(states, start_dim=1)
                    next_states = torch.flatten(next_states, start_dim=1)

                    q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states).max(dim=1)[0]
                        targets = rewards + gamma * next_q_values * (1 - dones)

                    # Loss and optimization
                    loss = torch.nn.MSELoss()(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if ep % target_update_freq == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            print(f"Episode {ep}, Reward: {ep_reward}")