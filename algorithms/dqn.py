# Have everything on torch instead of numpy to avoid compatibility issues.

import torch
import random
import pygame
import wandb
from datetime import datetime
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
    def __init__(self, env, enable_wandb=True):
        super().__init__(env)

        try:
            self.inputs = prod(self.env.obs_shape)
        except TypeError:
            self.inputs = self.env.obs_shape
            
        self.outputs = self.env.action_shape

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_network = QNetwork(self.inputs, self.outputs).to(self.device)
        self.target_network = QNetwork(self.inputs, self.outputs).to(self.device)

        self.enable_wandb = enable_wandb
        if enable_wandb == True:
            wandb.init()
            wandb.watch(self.policy_network , log="all")

    # some functions to consider
    def get_action(self, state):

        if self.infer == 0 and random.random() < self.epsilon:
            action = random.choice(self.env.action_space)
            action = torch.tensor(action, device=self.device)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy_network(state))

        return action

    def update_network(self):
        pass

    def train(self, num_episodes, early_stopping_config=None):
        if self.enable_wandb == True:
            wandb.init(project="pygameAI", name="CartPole_DQN")

        gamma = 0.99
        lr = 1e-3

        self.epsilon = 1.0 # for get_action()
        epsilon_decay = 0.998
        epsilon_min = 0.1
        target_update_freq = 10
        
        buffer_size = 12000
        batch_size = 32
        replay_buffer = ReplayBuffer(buffer_size)

        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.env.initiate_pygame(game_speed = 120)

        total_timesteps = 0

        rolling_rewards = []
        self.infer = 0
        if early_stopping_config is not None:
            early_stopping = True
            init_threshold, test_threshold, test_episodes = vars(early_stopping_config).values()
            print(init_threshold, test_threshold, test_episodes)
        else:
            early_stopping = False

        try:
            for ep in range(num_episodes):
                ep_reward = 0
                _, _, obs, done, _ = self.env.reset(render=True)

                state = torch.flatten(obs)

                while not done:
                    action = self.get_action(state)

                    reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)

                    next_state = torch.flatten(obs)

                    ep_reward += reward

                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state

                    total_timesteps += 1

                    if self.infer == 0 and len(replay_buffer) >= batch_size:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                        q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            next_q_values = self.target_network(next_states).max(dim=1)[0]
                            targets = rewards + gamma * next_q_values * (1 - dones)

                        # Loss and optimization
                        loss = torch.nn.MSELoss()(q_values, targets)

                        if self.enable_wandb:
                            wandb.log({"reward": reward, "loss": loss})

                        optimizer.zero_grad()
                        loss.backward()

                        # gradient clipping to avoid gradient explosion
                        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

                        optimizer.step()
                    
                        if (total_timesteps % 50 == 0):
                            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

                if ep % target_update_freq == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

                # print(f"Epsilon: {self.epsilon}")
                if self.infer == 0:
                    print(f"Episode {ep}, Reward: {ep_reward}")
                else:
                    print(f"Inferring Episode {ep}, Reward: {ep_reward}")

                if self.enable_wandb:
                    wandb.log({"episode": ep, "total_reward": ep_reward})

                rolling_rewards.append(ep_reward)
                if (len(rolling_rewards) > 10):
                    rolling_rewards = rolling_rewards[1:]
                    rolling_average = sum(rolling_rewards) / 10
                else:
                    rolling_average = None
                
                if early_stopping and rolling_average and rolling_average >= init_threshold and self.infer == 0:
                    self.infer = test_episodes

                if self.infer == 1 and rolling_average and rolling_average >= test_threshold:
                    print('test passed')
                    break
                
                self.infer = max(0, self.infer - 1)

        finally:
            if self.enable_wandb:
                wandb.finish()
            current_time = datetime.today().strftime('%Y_%m_%d_%H_%M_%S_')

            try:
                torch.save(self.policy_network.state_dict(), "saved_models/" + current_time + self.env.name + "_DQN.pth")
            except AttributeError:
                torch.save(self.policy_network.state_dict(), "saved_models/" + current_time + "_DQN.pth")

                
    def test(self, num_episodes):
        self.policy_network.load_state_dict(torch.load("policy_network_weights.pth"))

        self.epsilon = 0

        for ep in range(num_episodes):
            ep_reward = 0
            _, _, obs, done, _ = self.env.reset(render=True)

            state = torch.flatten(obs)

            while not done:
                action = self.get_action(state)

                reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)

                next_state = torch.flatten(obs)

                ep_reward += reward

                state = next_state


            print(f"Episode {ep}, Reward: {ep_reward}")