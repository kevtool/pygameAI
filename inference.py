import statistics
from games import PipeGame, FishingGame, FoodGame
from games.CartPole import CartPole
from algorithms.dqn import DQN

saved_model_path = "saved_models/2025_03_10_20_22_44_CartPole_DQN.pth"

game = CartPole(render_mode="human")
model = DQN(game, enable_wandb=False)
model.test(path=saved_model_path, num_episodes=10)

# saved_model_path = "saved_models/2025_03_10_15_23_53_PipeGame_DQN.pth"

# game = PipeGame(use_values_for_obs=True)
# model = DQN(game, enable_wandb=False)
# model.test(path=saved_model_path, num_episodes=10)
