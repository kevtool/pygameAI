import statistics
from games.PipeGame import PipeGame
from algorithms.dqn import DQN

game = PipeGame()

model = DQN(game)
model.train(10000)