import numpy as np
from utils import argsort

class Layer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    @staticmethod
    def setWeights(weights, biases):
        w = weights.copy()
        b = biases.copy()
        return Layer(w, b)

    @staticmethod
    def randomWeights(inputs, outputs):
        w = np.random.normal(0, 1, size=(inputs, outputs))
        b = np.random.normal(0, 1, size=(1, outputs))
        return Layer(w, b)

    def forward(self, input):
        # input: 1 * N
        return input @ self.weights + self.biases

class ReLu:
    def forward(self, input):
        return np.maximum(input, 0)

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.activation = ReLu()

    def create_layers(self, layerlist):
        self.layers = []
        if len(layerlist) == 0:
            self.layers.append(Layer.randomWeights(self.inputs, self.outputs))
            return
        
        self.layers.append(Layer.randomWeights(self.inputs, layerlist[0]))
        for i in range(0, len(layerlist) - 1):
            self.layers.append(Layer.randomWeights(layerlist[i], layerlist[i + 1]))
        self.layers.append(Layer.randomWeights(layerlist[-1], self.outputs))
    
    def mutate_layers(self):
        for layer in self.layers:
            # mutate weights
            shape = layer.weights.shape
            mutation = np.random.uniform(-0.05, 0.05, shape)
            layer.weights += (layer.weights * mutation)

            # mutate biases
            shape = layer.biases.shape
            mutation = np.random.uniform(-0.05, 0.05, shape)
            layer.biases += (layer.biases * mutation)

    @classmethod
    def create(cls, inputs, layerlist, outputs):
        obj = cls(inputs, outputs)
        obj.create_layers(layerlist)
        return obj

    @classmethod
    def mutate(cls, nn):
        obj = cls(nn.inputs, nn.outputs)
        obj.layers = [Layer.setWeights(l.weights, l.biases) for l in nn.layers]
        obj.mutate_layers()
        return obj

    @classmethod
    def copy(cls, nn):
        obj = cls(nn.inputs, nn.outputs)
        obj.layers = [Layer.setWeights(l.weights, l.biases) for l in nn.layers]
        return obj

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = self.activation.forward(x)

        return x

class Algorithm:
    def __init__(self, brains_per_gen):
        self.brains = [NeuralNetwork.create(inputs=4, layerlist=[3], outputs=1) for _ in range(brains_per_gen)]
        self.scores = []
    
    # changes: get the number of direction changes
    def record_score(self, index, score, changes):
        self.scores.append((index, score, changes))

    # get a list of how the descendants should be distributed among qualified brains
    def get_desc_list(self, num_of_descendants, qualified_brains):
        denom = sum(i ** -0.8 for i in range(1, qualified_brains+1))

        desc_float = [((i+1) ** -0.8) / denom * num_of_descendants for i in range(qualified_brains)]
        desc_int = [round(((i+1) ** -0.8) / denom * num_of_descendants) for i in range(qualified_brains)]
        desc_diff = [f - i for f, i in zip(desc_float, desc_int)]

        if (sum(desc_int) < num_of_descendants):
            desc_diff = argsort(desc_diff, reverse=True)
            for i in range(num_of_descendants - sum(desc_int)):
                desc_int[desc_diff[i]] += 1

        if (sum(desc_int) > num_of_descendants):
            desc_diff = argsort(desc_diff, reverse=False)
            for i in range(sum(desc_int) - num_of_descendants):
                desc_int[desc_diff[i]] -= 1

        assert sum(desc_int) == num_of_descendants
        return desc_int

    # create a new generation of brains and replace the old ones
    def create_new_gen(self, brains_per_gen):
        INDEX = 0

        num_of_descendants = int(brains_per_gen / 2)

        # sort by second element (score)
        self.scores = sorted(self.scores, key=lambda x: (x[1], x[2]), reverse=True)
        qualified_brains = int(min(sum(1 for tpl in self.scores if tpl[2] != 0), num_of_descendants / 2))

        # if there are no qualified brains, create new ones with random weights
        if qualified_brains == 0:
            new_brains = [NeuralNetwork.create(inputs=4, layerlist=[3], outputs=1) for _ in range(num_of_descendants)]
        else:
            desc_list = self.get_desc_list(num_of_descendants, qualified_brains)
            new_brains = []

            for i, descendants in enumerate(desc_list):
                index = self.scores[i][INDEX]
                brain_to_reproduce = self.brains[index]
                
                for j in range(descendants):
                    if j == 0:
                        new_brains.append(NeuralNetwork.copy(brain_to_reproduce))
                    else:
                        new_brains.append(NeuralNetwork.mutate(brain_to_reproduce))

        created_brains = [NeuralNetwork.create(inputs=4, layerlist=[3], outputs=1) for _ in range(brains_per_gen - num_of_descendants)]
        new_brains = new_brains + created_brains

        assert len(new_brains) == brains_per_gen
        self.brains = new_brains
        self.scores = []

    def print_brains_weights(self):
        for brain in self.brains:
            print(brain.layers[0].weights)