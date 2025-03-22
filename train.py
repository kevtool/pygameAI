import statistics
from games import PipeGame, FishingGame, FoodGame
from games.CartPole import CartPole
from algorithms.dqn import DQN
from config_classes import DQNConfig, EarlyStoppingConfig

# # CartPole Example

# game = CartPole()
# model = DQN(game, enable_wandb=False)

# config = DQNConfig()

# early_stopping_config = EarlyStoppingConfig(
#     init_threshold=400, 
#     init_episodes=10, 
#     test_threshold=500, 
#     test_episodes=10, 
#     cooldown=10
# )

# model.train(10000, config=config, early_stopping_config=early_stopping_config)

# # PipeGame Example

# game = PipeGame(use_values_for_obs=True)
# model = DQN(game, enable_wandb=False)
# model.train(10000, timesteps_per_decay=400, speed=300)

# # trying FoodGame

network_config = {
    "input_channels": 3,
    "input_size": (80, 45),  # (Height, Width)
    "conv_layers": [
        {"out_channels": 32, "kernel_size": 8, "stride": 4},
        {"out_channels": 64, "kernel_size": 4, "stride": 2},
        {"out_channels": 64, "kernel_size": 3, "stride": 1},
    ],
    "fc_units": 256,
    "num_actions": 4
}

game = FoodGame(use_values_for_obs=False)
model = DQN(game, network_config=network_config, enable_wandb=True)
model.train(100000, timesteps_per_decay=500, speed=120)