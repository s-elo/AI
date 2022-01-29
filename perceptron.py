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

        cur_epoch = 0
        for i in range(0, epoch):
            cur_epoch = cur_epoch + 1

            last_activated = self.forward_progress(
                input_data=input_data, activation='step')

            # get the error
            errors = input_label - last_activated

            # converged
            if np.all(errors == 0):
                break

            # only one layer
            weights = self.weights[0]
            # update weights (feature_size, neuron_num(1)) + (feature_size, sample_num) * (sample_num, 1)
            self.weights[0] = weights + learning_rate * \
                np.dot(input_data, errors.T)

        print(cur_epoch)
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

    def regression_LLS(self, input_data, input_labels):
        # (sample_num, feature_size)
        regression_matrix = input_data.T

        return np.dot(np.dot(np.linalg.inv(np.dot(regression_matrix.T, regression_matrix)), regression_matrix.T), input_labels.T)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def step(self, z):
        return 1 * (z >= 0)


if __name__ == '__main__':
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]).T
    Y = np.array([[0, 1, 1, 1]])
    # X = np.array([[1, 0], [1, 1]]).T
    # Y = np.array([[1, 0]])

    sp = Perceptron()

    # (input_feature, neuron_num)
    # sp.add_layer(weight_shape=(3, 2))
    # sp.add_layer(weight_shape=(2, 1))
    sp.add_layer(weight_shape=(3, 1))

    weights = sp.train(X, Y, learning_rate=0.001, epoch=100)
    print(weights)

    def draw_line(weights, input_data, input_labels, regression=False):
        if regression:
            bias = weights[0]
            slop = weights[1]
        else:
            bias = -weights[0] / weights[2]
            slop = -weights[1] / weights[2]

        plt.figure()
        plt.plot([-1, 5], [-1*slop + bias, 5*slop + bias])

        if regression:
            plt.scatter(input_data[1:], input_labels[0:], c='red', marker='x')
        else:
            plt.scatter(input_data[1:2], input_data[2:], c='red', marker='x')

    draw_line(weights.T[0], X, Y)

    regression_data = np.array([[1, 1], [1, 1.5], [1, 3], [1, 3.5], [1, 4]]).T
    regression_labels = np.array([[3.5, 2.5, 2, 1, 1]])

    regression_weights = sp.regression_LLS(regression_data, regression_labels)
    print(regression_weights.T[0])

    draw_line(regression_weights, regression_data,
              regression_labels, regression=True)

    plt.show()
