from numpy import exp, array, dot, random


class NeuralNetwork():
    def __init__(self):
        # inicializa gerador de numeros aleatorios
        random.seed(1)

        self.synaptic_weights = 2 * random.random((3, 2)) - 1
	self.synaptic_weights_end = 2 * random.random((2, 1)) - 1

    def __sigmoid(self, x):
        result = 1 / (1 + exp(-x))
        return result

    def __sigmoid_derivative(self, x):
        result = x * (1 - x)
        return result

    def think(self, inputs):
        result = self.__sigmoid(dot(inputs, self.synaptic_weights))
        return result

    def think_end(self, inputs):
        result = self.__sigmoid(dot(inputs, self.synaptic_weights_end))
        return result

    def train(self, training_set_inputs, training_set_outputs,
              number_of_training_interations):
        for interation in xrange(number_of_training_interations):

	    outputs = self.think(training_set_inputs)
        outputs_end = self.think_end(outputs)
        print "dentro do train:"
	    print "outputs", outputs.shape
	    print "outputs_end", outputs_end.shape
	    error_end = training_set_outputs - outputs_end
        delta_end = error_end * self.__sigmoid_derivative(outputs_end)

        error = dot(delta_end, self.synaptic_weights_end.T)
        delta = error * self.__sigmoid_derivative(outputs)

        adjustment_end = dot(outputs.T, delta_end)
        adjustment = dot(training_set_inputs.T, delta)

        self.synaptic_weights += adjustment
        self.synaptic_weights_end += adjustment_end

        print "outputs_end", outputs_end


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print "Inicializando pesos das entradas"
    print neural_network.synaptic_weights

    training_inputs = array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    training_outputs = array([[0], [1], [1], [0]])

    neural_network.train(training_inputs, training_outputs, 5000000)

    print "Novo pesos das entradas"
    print neural_network.synaptic_weights

    print "Testando com entradas: '1' e '1'"
    print neural_network.think_end(neural_network.think(training_inputs))
