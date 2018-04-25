from numpy import exp, array, dot, random


class NeuralNetwork():
    def __init__(self):
        # inicializa gerador de numeros aleatorios
       random.seed(1)

       self.synaptic_weights_1layer = 2 * random.random((3, 2)) - 1
       self.synaptic_weights_2layer = 2 * random.random((2, 1)) - 1

    def sigmoid(self, x):
        result = 1 / (1 + exp(-x))
        return result

    def sigmoid_derivative(self, x):
        result = x * (1 - x)
        return result

    def think(self, inputs):
        result = self.sigmoid(dot(inputs, self.synaptic_weights_1layer))
        return result

    def think_2layer(self, inputs):
        result = self.sigmoid(dot(inputs, self.synaptic_weights_2layer))
        return result

    def think_all(self, inputs):
        result = self.think_2layer(self.think(inputs))
        return result

    def train(self, training_set_inputs, training_set_outputs,
              number_of_training_interations):
        for interation in xrange(number_of_training_interations):
            outputs_1layer = self.think(training_set_inputs)
            outputs_2layer = self.think_2layer(outputs_1layer)

            error_2layer = training_set_outputs - outputs_2layer
            delta_2layer = error_2layer * self.sigmoid_derivative(outputs_2layer)

            error_1layer = dot(delta_2layer, self.synaptic_weights_2layer.T)
            delta_1layer = error_1layer * self.sigmoid_derivative(outputs_1layer)

            adjustment_2layer = dot(outputs_1layer.T, delta_2layer)
            adjustment_1layer = dot(training_set_inputs.T, delta_1layer)

            self.synaptic_weights_1layer += adjustment_1layer
            self.synaptic_weights_2layer += adjustment_2layer


if __name__ == "__main__":
    neural_network = NeuralNetwork()


    training_inputs = array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    training_outputs = array([[0], [1], [1], [0]])

    neural_network.train(training_inputs, training_outputs, 5)

    print "Testando com entradas 00 ; 01 ; 10 ; 11 :"
    print neural_network.think_all(training_inputs)
