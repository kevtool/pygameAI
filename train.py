import statistics
from games import PipeGame, FishingGame, FoodGame
from games.CartPole import CartPole
from algorithms.dqn import DQN
from config_classes import EarlyStoppingConfig

game = CartPole()

model = DQN(game, enable_wandb=False)


early_stopping_config = EarlyStoppingConfig(init_threshold=420, test_threshold=500, test_episodes=10)

model.train(10000, early_stopping_config=early_stopping_config)