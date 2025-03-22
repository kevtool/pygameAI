# Have everything on torch instead of numpy to avoid compatibility issues.

import torch
import random
import wandb
from datetime import datetime
from math import prod
from collections import deque
from algorithms.algorithm import Algorithm

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config=None):

        super().__init__()
        if config:
            layerlist = []
            in_channels = config["input_channels"]
            for conv in config["conv_layers"]:
                layerlist.append(torch.nn.Conv2d(in_channels, conv["out_channels"], conv["kernel_size"], conv["stride"]))
                layerlist.append(torch.nn.ReLU())
                in_channels = conv["out_channels"]
            
            self.cnn = torch.nn.Sequential(*layerlist)

            with torch.no_grad():
                sample_input = torch.zeros(1, config["input_channels"], *config["input_size"])
                sample_output = self.cnn(sample_input)
                conv_output_size = sample_output.view(1, -1).size(1)

            self.fc = torch.nn.Sequential(
                torch.nn.Linear(conv_output_size, config["fc_units"]),
                torch.nn.ReLU(),
                torch.nn.Linear(config["fc_units"], config["num_actions"])
            )

        else:
            self.cnn = torch.nn.Identity()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, action_dim)
            )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
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
    def __init__(self, env, network_config=None, enable_wandb=True):
        super().__init__(env)

        try:
            self.inputs = prod(self.env.obs_shape)
        except TypeError:
            self.inputs = self.env.obs_shape
            
        self.outputs = self.env.action_shape

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_network = QNetwork(self.inputs, self.outputs, config=network_config).to(self.device)
        self.target_network = QNetwork(self.inputs, self.outputs, config=network_config).to(self.device)

        self.enable_wandb = enable_wandb
        if enable_wandb == True:
            wandb.init()
            wandb.watch(self.policy_network , log="all")

    # some functions to consider
    def get_action(self, state):

        if  random.random() < self.epsilon and self.inferring == False:
            action = random.choice(self.env.action_space)
            action = torch.tensor(action, device=self.device)
        else:
            with torch.no_grad():
                state = state.unsqueeze(0)
                action = torch.argmax(self.policy_network(state))

        return action

    def update_network(self):
        pass

    def train(self, num_episodes, timesteps_per_decay=50, speed=120, config=None, early_stopping_config=None):
        if self.enable_wandb == True:
            wandb.init(project="pygameAI", name="CartPole_DQN")


        # DQN Config
        gamma = 0.99
        lr = 1e-3
        self.epsilon = 1.0 # for get_action()
        epsilon_decay = 0.998
        epsilon_min = 0.1
        target_update_freq = 10
        buffer_size = 12000
        batch_size = 32

        if config is not None:
            gamma, lr, epsilon_decay, epsilon_min, target_update_freq, _, _, timesteps_per_decay, buffer_size, batch_size = vars(config).values()


        # Early Stopping Config
        rolling_rewards = []
        self.inferring = False
        cooldown = 0
        if early_stopping_config is not None:
            early_stopping = True
            init_threshold, init_episodes, self.test_threshold, self.test_episodes, self.cooldown = vars(early_stopping_config).values()
        else:
            early_stopping = False

        # Set up
        replay_buffer = ReplayBuffer(buffer_size)

        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.env.initiate_pygame(game_speed = speed)

        total_timesteps = 0

        # Main Loop
        try:
            for ep in range(num_episodes):
                ep_reward = 0
                _, _, obs, done, _ = self.env.reset(render=True)

                # state = torch.flatten(obs)
                state = obs

                while not done:
                    action = self.get_action(state)

                    reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)

                    # next_state = torch.flatten(obs)
                    next_state = obs

                    ep_reward += reward

                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state

                    total_timesteps += 1

                    if len(replay_buffer) >= batch_size:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                        q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            next_q_values = self.target_network(next_states).max(dim=1)[0]
                            targets = rewards + gamma * next_q_values * (1 - dones)

                            if torch.isnan(q_values).any() or torch.isnan(targets).any():
                                print("NaN encountered")

                        # Loss and optimization
                        loss = torch.nn.MSELoss()(q_values, targets)

                        if self.enable_wandb:
                            wandb.log({"reward": reward, "loss": loss, "epsilon": self.epsilon})

                        optimizer.zero_grad()
                        loss.backward()

                        # gradient clipping to avoid gradient explosion
                        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

                        optimizer.step()
                    
                        if (total_timesteps % timesteps_per_decay == 0):
                            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

                if ep % target_update_freq == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

                # print(f"Epsilon: {self.epsilon}")
                print(f"Episode {ep}, Reward: {ep_reward}")
                
                if self.enable_wandb:
                    wandb.log({"episode": ep, "total_reward": ep_reward})


                # Early Stopping Code
                if early_stopping:
                    rolling_rewards.append(ep_reward)
                    if (len(rolling_rewards) > init_episodes):
                        rolling_rewards = rolling_rewards[1:]
                        rolling_average = sum(rolling_rewards) / init_episodes
                    else:
                        rolling_average = None
                
                    if rolling_average is not None and rolling_average >= init_threshold and cooldown == 0:
                        p = self.infer()
                        if p:
                            print("Test passed")
                            break
                        else:
                            cooldown = self.cooldown

                cooldown = max(0, cooldown - 1)

        finally:
            if self.enable_wandb:
                wandb.finish()
                current_time = datetime.today().strftime('%Y_%m_%d_%H_%M_%S_')

                try:
                    torch.save(self.policy_network.state_dict(), "saved_models/" + current_time + self.env.name + "_DQN.pth")
                except AttributeError:
                    torch.save(self.policy_network.state_dict(), "saved_models/" + current_time + "_DQN.pth")

    # This is used by training to determine early stopping / see non-exploration results mid training
    def infer(self):
        test_rewards = []

        self.inferring = True

        for ep in range(self.test_episodes):
            ep_reward = 0
            _, _, obs, done, _ = self.env.reset(render=True)

            # state = torch.flatten(obs)
            state = obs

            while not done:
                action = self.get_action(state)

                reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)

                # next_state = torch.flatten(obs)
                next_state = obs

                ep_reward += reward

                state = next_state


            print(f"Inferring Episode {ep}, Reward: {ep_reward}")
            test_rewards.append(ep_reward)

        if sum(test_rewards) / self.test_episodes >= self.test_threshold:
            return True
        
        return False

    # This should be called outside    
    def test(self, path, num_episodes=10):
        self.policy_network.load_state_dict(torch.load(path))

        self.epsilon = 0

        self.env.initiate_pygame(game_speed = 60)

        for ep in range(num_episodes):
            ep_reward = 0
            _, _, obs, done, _ = self.env.reset(render=True)

            # state = torch.flatten(obs)
            state = obs

            while not done:
                action = self.get_action(state)

                reward, _, obs, done, info = self.env.step(action, mode='ai', render=True)

                # next_state = torch.flatten(obs)
                next_state = obs

                ep_reward += reward

                state = next_state

            print(f"Episode {ep}, Reward: {ep_reward}")