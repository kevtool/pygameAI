import statistics
from games.PipeGame import PipeGame
from games.FishingGame import FishingGame
from algorithms.dqn import DQN

game = PipeGame()

model = DQN(game)
model.train(10000)