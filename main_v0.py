from numpy import exp, array, dot, random


class NeuralNetwork():
    def __init__(self):
        # inicializa gerador de numeros aleatorios
        random.seed(1)

        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        result = 1 / (1 + exp(-x))
        return result

    def __sigmoid_derivative(self, x):
        result = x * (1 - x)
        return result

    def think(self, inputs):
        result = self.__sigmoid(dot(inputs, self.synaptic_weights))
        return result

    def train(self, training_set_inputs, training_set_outputs,
              number_of_training_interations):
        for interation in xrange(number_of_training_interations):
            outputs = self.think(training_set_inputs)
            error = training_set_outputs - outputs
            adjustment = dot(training_set_inputs.T, error *
                             self.__sigmoid_derivative(outputs))
            self.synaptic_weights += adjustment


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print "Inicializando pesos das entradas"
    print neural_network.synaptic_weights

    training_inputs = array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
                             [1, 1, 1]])
    training_outputs = array([[0], [1], [1], [0], [1]])

    neural_network.train(training_inputs, training_outputs, 1000000)

    print "Novo pesos das entradas"
    print neural_network.synaptic_weights

    print "Testando com entradas: '1' e '1'"
    print neural_network.think(array([0, 0, 0]))
