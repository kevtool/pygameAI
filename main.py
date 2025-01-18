import statistics
from game import Game
from algorithms.dqn import DQN

game = Game()

model = DQN(game)
model.train(1)