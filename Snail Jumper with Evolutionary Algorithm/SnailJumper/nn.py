import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.num_of_layers=len(layer_sizes)
        self.layer_sizes=layer_sizes
        self.weights=[]
        self.biases=[]
        for i in range(1,self.num_of_layers):
            self.weights.append(np.random.normal(size=(layer_sizes[i],layer_sizes[i-1])))
            self.biases.append(np.zeros((layer_sizes[i],1)))
        
        self.layers=[]
        for i in range(0,self.num_of_layers):
            self.layers.append(np.zeros((layer_sizes[i],1)))

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return 1.0 / (1 + np.exp(-1 * x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        self.layers[0]=x
        for i in range(0,self.num_of_layers-1):
            self.layers[i+1]=self.weights[i] @ self.layers[i] + self.biases[i]
            self.layers[i+1]=self.activation(self.layers[i+1])
        
        return self.layers[self.num_of_layers-1]
