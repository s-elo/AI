import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, input_shape):
        self.feature_size = input_shape[0]
        self.sample_num = input_shape[1]

        self.layers = []
        self.activations = {
            'sigmoid': self.sigmoid,
            'step': self.step,
            'linear': self.linear
        }

    def add_layer(self, neuron_num, activation='sigmoid'):
        layer_num = len(self.layers)

        if layer_num == 0:
            # (feature_size, neuron_num)
            self.layers.append({
                'weight': np.random.randn(
                    self.feature_size + 1, neuron_num) * 0.01,
                'activation': self.activations[activation]
            })
        else:
            # (feature_size, neuron_num)
            self.layers.append({
                'weight': np.random.randn(
                    self.layers[layer_num - 1]['weight'].shape[1] + 1, neuron_num) * 0.01,
                'activation': self.activations[activation]
            })

    def forward_progress(self, input_data):
        # add bias
        prev_activated = np.append(
            input_data, np.ones(shape=(1, self.sample_num)), axis=0)

        # for backward progress
        # the activated is the input_data initially
        self.cache = [(input_data, input_data)]

        # forward progress
        for layer in self.layers:
            weight = layer['weight']
            activation = layer['activation']

            # (neuron_num, feature_size)*(feature_size, sample_num)
            local_field = np.dot(weight.T, prev_activated)
            activated = activation(local_field)

            assert(activated.shape == (weight.shape[1], self.sample_num))

            prev_activated = np.append(
                activated, np.ones(shape=(1, self.sample_num)), axis=0)

            self.cache.append((local_field, activated))

        # (output_size(neuron_num), sample_num)
        return activated

    def back_progress(self):
        for layer_index in range(len(self.layers) - 1, -1, -1):
            print(layer_index)

    # single perceptron for classification
    def train(self, input_data, input_label, epochs=10, learning_rate=1):
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
                np.dot(np.append(
                    input_data, np.ones(shape=(1, self.sample_num)), axis=0), learning_rate * errors.T)

        print(cur_epoch)
        return self.layers[0]['weight']

    def predict(self, input_data):
        return self.forward_progress(input_data)

    def regression_LLS(self, input_data, input_labels):
        # (sample_num, feature_size)
        regression_matrix = np.append(
            input_data, np.ones(shape=(1, self.sample_num)), axis=0).T

        return np.dot(np.dot(np.linalg.inv(np.dot(regression_matrix.T, regression_matrix)), regression_matrix.T), input_labels.T)

    def regression_LMS(self, input_data, input_labels, epochs=100, learning_rate=0.01):
        input_data = np.append(
            input_data, np.ones(shape=(1, self.sample_num)), axis=0)

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

            w_cache.append(weights[0])
            b_cache.append(weights[1])

        return (weights, w_cache, b_cache)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def step(self, z):
        return 1 * (z >= 0)

    def linear(self, z):
        return z


if __name__ == '__main__':
    p = Perceptron(input_shape=(2, 5))
    p.add_layer(neuron_num=2)
    p.add_layer(neuron_num=1)
    p.back_progress()

    arr1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]).T
    arr2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).T
    print((arr1 - arr2))
    print(np.append(arr1, np.ones(shape=(1, 4)), axis=0))
