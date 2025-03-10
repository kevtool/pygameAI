import statistics
from games import PipeGame, FishingGame, FoodGame
from games.CartPole import CartPole
from algorithms.dqn import DQN

game = CartPole()

model = DQN(game)
model.train(10000)