September 10 update:
Reorganized code to make it more generalized and usable.
Bug fixed: when pygame is run, the first iteration of the game has delayed times for pipe entries, which causes the first iteration to always have a higher score.

________
#### Summary

The first part of the project is to build the game itself. In the game, the player simply controls the ship to go up or down, and let it fly through the pipes without colliding. The ship dies if it collides with the pipes. The player generates score as long as the ship is alive. I wrote the game on pygame which allows me to directly take the game info and score into the AI as input.

The AI is a genetic algorithm that generates many weights and biases for the neural network system, then choose the best performing sets of weights and biases to continue with. For the neural network, I chose to feed it 4 inputs: the player's y position (1 input), and the x and y positions of the nearest pair of pipes (3 inputs). If the output of the neural network is positive, the brain the commands the ship to fly upwards, and if the output is negative, the brain commands the ship to fly downwards. I figured that since the output is only going to be positive or negative, and that determines the action of the AI, there's no need to do logistic regression in the last layer. Whether or not this works remains to be seen.

Our AI is now ready to play the game (even though I haven't implemented the learning part yet). I quickly noticed that while some iterations of the brain will commmand the ship to go up or down as new information on the pipes come in, most iterations will only command the ship to go either up or down, with no change in direction. For some of these iterations, while the output of the neural network changes, it is not enough to change the sign of the output (positive or negative), and therefore the ship does not change its direction. This causes the vast majority of brains to not be able to survive even the first pair of pipes.

To solve this problem, I need to get rid of these brains that do not change the ship's direction. For those brains that do change the ship's direction, I assign a higher score than brains that do not change direction, so that the brains that change direction will be selected to test in the next generation. This way, I can identify weights and biases that are useful, even though almost every brain cannot pass even the first pair of pipes at this stage.