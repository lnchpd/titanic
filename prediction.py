import pandas as pd
import numpy as np
import matplotlib as plt

import sklearn.metrics as skmetrics

from clean_data import targets, test_targets, inputs, test_inputs

np.random.seed(1234)

#standard sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#the derivative of the sigmoid function, for use in gradient descent
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Neural Network object
class NeuralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        #assign random numbers as initial weights with len(input) = rows, len(output) = columns
        self.weights_i2h = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
        self.weights_h2o = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))

    #define hidden node function as a sigmoid
    def hidden_f(self, x):
        return sigmoid(x)

    def hidden_fp(self, x):
        return sigmoid_prime(x)

    #define output node function as a sigmoid
    def output_f(self, x):
        #return x
        return sigmoid(x)

    def output_fp(self, x):
        #return 1
        return sigmoid_prime(x)

    #send a forward pass through the network and return each step, along with error
    def forward_pass(self, record_input, record_target):

        #input to the hidden node is the weights * inputs, respectively
        #transpose the weights since record_input is simply a 1x14 array
        #input results in a 1x2 array
        self.hidden_in = np.dot(self.weights_i2h.T, record_input)

        #apply the activation function of the hidden node
        self.hidden_out = self.hidden_f(self.hidden_in)

        #input to the output node is the weights * hidden_out; once again must transpose
        self.output_in = np.dot(self.weights_h2o.T, self.hidden_out)

        #apply the activation function of the output node
        self.output_out = self.output_f(self.output_in)

        #calculate error
        self.error = record_target - self.output_out

        #calculate the squared error
        self.sse = 0.5 * (self.error **2) / n_records

    #train the network on a single record with a forward pass and a backpropogation
    def train(self, record_input, record_target):
        self.forward_pass(record_input, record_target)

        #determine the effect of inputs to the output node on the error
        output_error = self.error * self.output_fp(self.output_in)

        #determine the effect of inputs to the hidden node on the error (using chain rule)
        hidden_error = output_error * self.weights_h2o.T * self.hidden_fp(self.hidden_in)

        #update the weights incrementally
        self.weights_h2o += self.learning_rate * self.hidden_out[:,None] * output_error
        self.weights_i2h += self.learning_rate * record_input[:,None] * hidden_error

#define hyperparameters for the neural network
LEARNING_RATE = .1
EPOCHS = 50
HIDDEN_NODES = 7

#determine number of records and number of inputs from the data
n_records, n_inputs = inputs.shape
n_targets = 1  #targets.shape[1] - doesn't work?
n_test_records = test_inputs.shape[0]

#define the neural network
nn = NeuralNetwork(n_inputs, HIDDEN_NODES, n_targets, LEARNING_RATE)

def run():
    #for each epoch
    for i in range(EPOCHS):

        error_rate = 0

        #for each record in the training set
        for x, y in zip(inputs.values, targets):

            #train the model and record the error rate
            nn.train(x, y)
            error_rate += nn.sse

        print(error_rate)

def test():

    test_output = []

    #for each record in the test set
    for a, b in zip(test_inputs.values, test_targets):

        #run a forward pass through the network
        nn.forward_pass(a, b)

        #round to 0 or 1 and record output for accuracy test
        test_output.append(round(nn.output_out[0],0))

    print('Accuracy: ', skmetrics.accuracy_score(test_targets, test_output))
    print(skmetrics.confusion_matrix(test_targets, test_output))

run()
test()
