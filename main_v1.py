from numpy import exp, zeros, dot, random, tanh, outer, mean, log
import time
# 2 layer neural neural network

# variables
# number of hidden neurons
n_hidden = 10
# number of inputs
n_in = 10
# number outputs
n_out = 10
# number of sample data (very big number to have a good range of training)
n_samples = 300

# hyperparametters
# step of learning rate
learning_rate = 0.01
momentum = 0.9

# seed random the same number every time run the code
# non deterministc sedding
random.seed(0)

# first activation function


def sigmoid(x):  # first activation function, sigmodi calc
    result = 1.0 / (1.0 + exp(-x))
    return result


def tanh_prime(x):  # second activation function, hiperbolic tangent element
    result = 1 - tanh(x) ** 2
    return result


def train(input_data, transpose, layerA, layerB, biasesA, biasesB):

    # foward propagation -- matrix multiply + biases
    A = dot(input_data, layerA) + biasesA
    # tangent of A
    Z = tanh(A)
    # matrix multiply + biases
    B = dot(Z, layerB) + biasesB
    # sigmoid of B
    Y = sigmoid(B)

    # backward propagation -- apli the sigmoid
    eA = Y - transpose
    eB = tanh_prime(A) + dot(layerB, eA)

    # predict loss
    dA = outer(Z, eA)
    dB = outer(input_data, eB)

    # cross entropy
    loss = -mean(transpose * log(Y) + (1 - transpose) * log(1 - Y))

    # return loss value, delta values, erros values for eatch layer
    return loss, (dA, dB, eA, eB)


def predict(input_data, layerA, layerB, biasesA, biasesB):
    A = dot(input_data, layerA) + biasesA
    B = dot(tanh(A), layerB) + biasesB
    return (sigmoid(B) > 0.5).astype(int)


# creating layers and biases


layerA = random.normal(scale=0.1, size=(n_in, n_hidden))
layerB = random.normal(scale=0.1, size=(n_in, n_out))

biasesA = zeros(n_hidden)
biasesB = zeros(n_out)

# short cut for writing the next code
params = [layerA, layerB, biasesA, biasesB]

X = random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1


# training
for epoch in range(100):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()
    # for each data point will update the synaptic_weights
    for i in range(X.shape[0]):
        # training for each time
        loss, grad = train(X[i], T[i], *params)
        # update loss
        for j in range(len(params)):
            params[j] -= upd[j]
        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] * momentum * upd[j]
        err.append(loss)
    print('Epoch: %d, Loss %.8f, Time: %.4f s' % (epoch, mean(err),
          time.clock() - t0))

# ruining this..

input_data = random.binomial(1, 0.5, n_in)
print('XOR prediction')
print(input_data)
print(predict(input_data, *params))
