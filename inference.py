import statistics
from games import PipeGame, FishingGame, FoodGame
from games.CartPole import CartPole
from algorithms.dqn import DQN

game = CartPole(render_mode="human")
model = DQN(game, enable_wandb=False)
model.test(10)