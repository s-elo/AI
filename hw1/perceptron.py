from copy import copy
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
            'identity': self.identity,
            'relu': self.relu
        }

        self.backwards = {
            'sigmoid': self.sigmoid_backward,
            'step': self.step_backward,
            'identity': self.identity_backward,
            'relu': self.relu_backward
        }

    def add_layer(self, neuron_num, activation='sigmoid'):
        layer_num = len(self.layers)

        if layer_num == 0:
            # (feature_size, neuron_num)
            self.layers.append({
                'weight': np.random.randn(
                    self.feature_size + 1, neuron_num) * 0.01,
                'activation': self.activations[activation],
                'backward': self.backwards[activation]
            })
        else:
            # (feature_size, neuron_num)
            self.layers.append({
                'weight': np.random.randn(
                    self.layers[layer_num - 1]['weight'].shape[1] + 1, neuron_num) * 0.01,
                'activation': self.activations[activation],
                'backward': self.backwards[activation]
            })

    def forward_progress(self, input_data):
        # add bias
        prev_activated = np.append(
            input_data, np.ones(shape=(1, self.sample_num)), axis=0)

        # for backward progress
        # the activated is the input_data initially
        self.cache = [{
            'local_field': prev_activated,
            'activated': prev_activated
        }]

        # forward progress
        for layer in self.layers:
            weight = layer['weight']
            activation = layer['activation']

            # (neuron_num, prev_neuron + 1)*(prev_neuron + 1, sample_num)
            local_field = np.dot(weight.T, prev_activated)
            activated = activation(local_field)

            assert(activated.shape == (weight.shape[1], self.sample_num))

            prev_activated = np.append(
                activated, np.ones(shape=(1, self.sample_num)), axis=0)

            self.cache.append({
                'local_field': local_field,
                'activated': prev_activated
            })

        # (output_size(neuron_num), sample_num)
        return activated

    def backward_progress(self, input_label, learning_rate):
        dw = []

        cache_len = len(self.cache)

        # [cache_len - 1, 1]
        for index in range(cache_len - 1, 0, -1):
            local_field = self.cache[index]['local_field']
            # (prev_neuron + 1, sample)
            prev_activated = self.cache[index - 1]['activated']

            if index == cache_len - 1:
                # output layer
                # (neuron, sample)
                output = self.layers[index - 1]['activation'](local_field)
                assert(output.shape == input_label.shape)
                error = input_label - output

                # (neuron, sample) (neuron, sample)
                delta = error * self.layers[index - 1]['backward'](local_field)
            else:
                # (neuron, next_neuron)
                weights = self.layers[index]['weight']
                # hidden layers
                # (neuron, next_neuron) (next_neuron, sample) = (neuron, sample)
                # remove the bias
                error = np.dot(weights[0:weights.shape[0] - 1], delta)

                # (neuron, sample) (neuron, sample)
                delta = error * self.layers[index - 1]['backward'](local_field)

            # (prev_neuron + 1, sample) (neuron, sample).T = (prev_neuron + 1, neuron)
            update = np.dot(prev_activated, learning_rate * delta.T)
            assert(update.shape == self.layers[index - 1]['weight'].shape)

            dw.insert(0, update)

        return dw

    def train(self, input_data, input_label, epochs=10, learning_rate=1):
        cur_epoch = 0

        for i in range(0, epochs):
            cur_epoch = cur_epoch + 1

            last_activated = self.forward_progress(
                input_data=input_data)

            dw = self.backward_progress(
                input_label=input_label, learning_rate=learning_rate)

            allZero = True
            for w in dw:
                if np.any(w != 0):
                    allZero = False
                    break

            if allZero:
                break

            # update weights
            for layer_idx in range(0, len(dw)):
                assert(self.layers[layer_idx]
                       ['weight'].shape == dw[layer_idx].shape)
                self.layers[layer_idx]['weight'] = self.layers[layer_idx]['weight'] + dw[layer_idx]

            output = self.forward_progress(
                input_data=input_data)

            if np.all((input_label - output) == 0):
                break

        print(f'epochs: {cur_epoch}')
        # return self.layers[0]['weight']
        return self.weights

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

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer['weight'])
        return weights

    def relu(self, z):
        return np.maximum(0, z)

    def relu_backward(self, z):
        x = np.array(z, copy=True)
        x[x <= 0] = 0
        x[x > 0] = 1

        return x

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_backward(self, z):
        s = self.sigmoid(z)

        return s * (1 - s)

    def step(self, z):
        return 1 * (z >= 0)

    def step_backward(self, z):
        return 1

    def identity(self, z):
        return z

    def identity_backward(self, z):
        return 1


# test
if __name__ == '__main__':
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, label="cos", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('sin & cos')
    plt.legend()

    x1 = np.reshape(x, (1, 60))
    y1 = np.reshape(y1, (1, 60))

    # (feature, sample)
    X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    Y = np.array([[0, 1, 0, 1]])

    regression_data = np.array(
        [[0], [0.8], [1.6], [3], [4], [5]]).T
    regression_labels = np.array([[0.5, 1, 4, 5, 6, 8]])

    # classifier = Perceptron(input_shape=regression_data.shape)
    classifier = Perceptron(input_shape=x1.shape)

    # (input_feature, neuron_num)
    classifier.add_layer(neuron_num=3, activation='relu')
    classifier.add_layer(neuron_num=1, activation='identity')

    # weights = classifier.train(
    #     regression_data, regression_labels, learning_rate=0.001, epochs=100)
    weights = classifier.train(
        x1, y1, learning_rate=0.001, epochs=100)
    # print(weights)

    # output = classifier.predict(regression_data)
    output = classifier.predict(x1)

    plt.figure()
    # plt.plot(regression_data[0], output[0])
    # plt.scatter(regression_data, regression_labels[0:], c='red', marker='x')

    plt.plot((x1)[0], output[0])

    # arr1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]).T
    # arr2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).T
    # print((arr1 - arr2))
    # print(np.append(arr1, np.ones(shape=(1, 4)), axis=0))

    # print(arr1 * arr2)
    plt.show()
