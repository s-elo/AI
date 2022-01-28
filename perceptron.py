import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self):
        self.weights = []
        self.activations = {
            'sigmoid': self.sigmoid,
            'step': self.step
        }

    def add_layer(self, weight_shape):
        # (feature_size, neuron_num)
        self.weights.append(np.random.randn(
            weight_shape[0], weight_shape[1]) * 0.01)

    # single perceptron for classification
    def train(self, input_data, input_label, epoch=10, learning_rate=1):
        feature_size = input_data[0]
        sample_num = input_data[1]

        for i in range(0, epoch):
            last_activated = self.forward_progress(
                input_data=input_data, activation='step')

            # get the error
            errors = input_label - last_activated
            print(errors)
            # converged
            if np.all(errors == 0):
                return self.weights[0]

            # only one layer
            weights = self.weights[0]
            # update weights (feature_size, neuron_num(1)) + (feature_size, sample_num) * (sample_num, 1)
            self.weights[0] = weights + learning_rate * \
                np.dot(input_data, errors.T)

        return self.weights[0]

    def forward_progress(self, input_data, activation='sigmoid'):
        # for backward progress
        self.cache = []

        # forward progress
        for weight in self.weights:
            # (neuron_num, feature_size)*(feature_size, sample_num)
            local_field = np.dot(weight.T, input_data)
            activated = self.activations[activation](local_field)

            assert(activated.shape == (weight.shape[1], input_data.shape[1]))

            input_data = activated

            self.cache.append((local_field, activated))

        # (output_size(neuron_num), sample_num)
        return activated

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def step(self, z):
        return 1 * (z >= 0)


if __name__ == '__main__':
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]).T
    Y_AND = np.array([[0, 0, 1, 0]])

    print(X)

    sp = Perceptron()

    # (input_feature, neuron_num)
    # sp.add_layer(weight_shape=(3, 2))
    # sp.add_layer(weight_shape=(2, 1))
    sp.add_layer(weight_shape=(3, 1))
    print(sp.weights)
    weights = sp.train(X, Y_AND)
    print(weights)

    def get_points(weights):
        bias = -weights[0] / weights[2]
        slop = -weights[1] / weights[2]

        return ([-1, 2], [-1*slop + bias, 2*slop + bias])

    p1, p2 = get_points(weights)

    plt.plot(p1, p2)
    plt.scatter([0, 0, 1], [0, 1, 0], c='red', marker='x')
    plt.scatter([1], [1], c='green', marker='x')
    plt.show()
