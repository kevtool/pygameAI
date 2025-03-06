import statistics
from games import PipeGame, FishingGame
from algorithms.dqn import DQN

game = FishingGame()

model = DQN(game)
model.train(10000)