import statistics
from games.PipeGame import PipeGame
from games.FishingGame import FishingGame
from algorithms.dqn import DQN

game = FishingGame()

model = DQN(game)
model.train(10000)