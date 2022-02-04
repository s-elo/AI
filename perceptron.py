import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self):
        self.layers = []
        self.activations = {
            'sigmoid': self.sigmoid,
            'step': self.step,
            'linear': self.linear
        }

    def add_layer(self, weight_shape, activation='sigmoid'):
        # (feature_size, neuron_num)
        self.layers.append({
            'weight': np.random.randn(
                weight_shape[0], weight_shape[1]) * 0.01,
            'activation': self.activations[activation]
        })

    # single perceptron for classification
    def train(self, input_data, input_label, epochs=10, learning_rate=1):
        feature_size = input_data[0]
        sample_num = input_data[1]

        cur_epoch = 0
        for i in range(0, epochs):
            cur_epoch = cur_epoch + 1

            last_activated = self.forward_progress(
                input_data=input_data)

            # get the error
            errors = input_label - last_activated

            # converged
            if np.all(errors == 0):
                break

            # only one layer
            weights = self.layers[0]['weight']

            # update weights (feature_size, neuron_num(1)) + (feature_size, sample_num) * (sample_num, 1)
            self.layers[0]['weight'] = weights + \
                np.dot(input_data, learning_rate * errors.T)

        print(cur_epoch)
        return self.layers[0]['weight']

    def forward_progress(self, input_data):
        # for backward progress
        self.cache = []

        # forward progress
        for layer in self.layers:
            weight = layer['weight']
            activation = layer['activation']

            # (neuron_num, feature_size)*(feature_size, sample_num)
            local_field = np.dot(weight.T, input_data)
            activated = activation(local_field)

            assert(activated.shape == (weight.shape[1], input_data.shape[1]))

            input_data = activated

            self.cache.append((local_field, activated))

        # (output_size(neuron_num), sample_num)
        return activated

    def predict(self, input_data):
        return self.forward_progress(input_data)

    def regression_LLS(self, input_data, input_labels):
        # (sample_num, feature_size)
        regression_matrix = input_data.T
        
        return np.dot(np.dot(np.linalg.inv(np.dot(regression_matrix.T, regression_matrix)), regression_matrix.T), input_labels.T)

    def regression_LMS(self, input_data, input_labels, epochs=100, learning_rate=0.01):
        # (feature_size,1)
        weights = np.random.randn(
            input_data.shape[0], 1) * 0.01

        w_cache = []
        b_cache = []

        for i in range(0, epochs):
            # (1, sample_num)
            local_field = np.dot(weights.T, input_data)

            errors = input_labels - local_field

            weights = weights + np.dot(input_data, learning_rate * errors.T)

            b_cache.append(weights[0])
            w_cache.append(weights[1])

        return (weights, w_cache, b_cache)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def step(self, z):
        return 1 * (z >= 0)

    def linear(self, z):
        return z


if __name__ == '__main__':
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]).T
    Y = np.array([[0, 1, 1, 1]])
    # X = np.array([[1, 0], [1, 1]]).T
    # Y = np.array([[1, 0]])

    classifier = Perceptron()

    # (input_feature, neuron_num)
    # sp.add_layer(weight_shape=(3, 2))
    # sp.add_layer(weight_shape=(2, 1))
    classifier.add_layer(weight_shape=(3, 1), activation='step')

    weights = classifier.train(X, Y, learning_rate=0.001, epochs=100)
    # print(weights)

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

    # draw_line(weights.T[0], X, Y)

    regression_data = np.array(
        [[1, 0], [1, 0.8], [1, 1.6], [1, 3], [1, 4], [1, 5]]).T
    regression_labels = np.array([[0.5, 1, 4, 5, 6, 8]])

    regression = Perceptron()

    LLS_weights = regression.regression_LLS(regression_data, regression_labels)
    print(LLS_weights)

    LMS_weights, w_cache, b_cache = regression.regression_LMS(
        regression_data, regression_labels, learning_rate=0.01)
    print(LMS_weights)

    draw_line(LLS_weights, regression_data,
              regression_labels, regression=True)

    draw_line(LMS_weights, regression_data,
              regression_labels, regression=True)

    plt.figure()
    plt.plot(np.arange(0, 100, 1), w_cache)
    plt.plot(np.arange(0, 100, 1), b_cache)
    plt.legend(['w', 'b'])

    regression.add_layer(weight_shape=(2, 1), activation='linear')
    LMS_w = regression.train(
        regression_data, regression_labels, epochs=100, learning_rate=0.01)
    print(LMS_w)

    plt.show()
