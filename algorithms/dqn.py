# Have everything on torch instead of numpy to avoid compatibility issues.

import torch
import random
from math import prod
from collections import deque
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.uint8, device=self.device) 
        )

    def __len__(self):
        return len(self.buffer)

class DQN(Algorithm):
    def __init__(self, env):
        super().__init__(env)

        self.inputs = prod(self.env.obs_shape)
        self.outputs = self.env.action_shape

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_network = QNetwork(self.inputs, self.outputs).to(self.device)
        self.target_network = QNetwork(self.inputs, self.outputs).to(self.device)

    # some functions to consider
    def get_action(self, state):

        if random.random() < self.epsilon:
            action = random.choice(self.env.action_space)
            action = torch.tensor(action, device=self.device)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy_network(state))

        return action

    def update_network(self):
        pass

    def train(self, num_episodes):
        gamma = 0.99
        lr = 1e-3

        self.epsilon = 1.0 # for get_action()
        epsilon_decay = 0.995
        epsilon_min = 0
        target_update_freq = 10
        
        buffer_size = 12000
        batch_size = 32
        replay_buffer = ReplayBuffer(buffer_size)

        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.env.initiate_pygame()

        for ep in range(num_episodes):
            ep_reward = 0
            _, _, obs, done, _ = self.env.reset(render=True)

            # if performance is bad, test execution time with and without clone
            state = torch.flatten(obs)

            steps = 0

            while not done:
                action = self.get_action(state)

                reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)

                next_state = torch.flatten(obs)

                ep_reward += reward

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

                if len(replay_buffer) >= batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                    q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states).max(dim=1)[0]
                        targets = rewards + gamma * next_q_values * (1 - dones)

                    # Loss and optimization
                    loss = torch.nn.MSELoss()(q_values, targets)
                    
                    steps += 1
                    if steps % 50 == 0:
                        print(f"Loss: {loss.item()}")

                    optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping to avoid gradient explosion
                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

                    optimizer.step()
                
            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

            if ep % target_update_freq == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            print(f"Epsilon: {self.epsilon}")
            print(f"Episode {ep}, Reward: {ep_reward}")