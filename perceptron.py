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

        self.backwards = {
            'sigmoid': self.sigmoid_backward,
            'step': self.step_backward,
            'linear': self.linear_backward
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

            if np.all(np.array(dw) == 0):
                break

            # update weights
            for layer_idx in range(0, len(dw)):
                assert(self.layers[layer_idx]
                       ['weight'].shape == dw[layer_idx].shape)
                self.layers[layer_idx]['weight'] = self.layers[layer_idx]['weight'] + dw[layer_idx]

            # get the error
            # errors = input_label - last_activated

            # converged
            # if np.all(errors == 0):
            #     break

            # only one layer
            # weights = self.layers[0]['weight']

            # update weights (feature_size, neuron_num(1)) + (feature_size, sample_num) * (sample_num, 1)
            # self.layers[0]['weight'] = weights + \
            #     np.dot(np.append(
            #         input_data, np.ones(shape=(1, self.sample_num)), axis=0), learning_rate * errors.T)

        print(cur_epoch)
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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_backward(self, z):
        s = self.sigmoid(z)

        return s * (1 - s)

    def step(self, z):
        return 1 * (z >= 0)

    def step_backward(self, z):
        return 1

    def linear(self, z):
        return z

    def linear_backward(self, z):
        return 1


if __name__ == '__main__':
   # (feature, sample)
    X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    Y = np.array([[0, 1, 0, 1]])

    classifier = Perceptron(input_shape=X.shape)

    # (input_feature, neuron_num)
    classifier.add_layer(neuron_num=2, activation='sigmoid')
    classifier.add_layer(neuron_num=1, activation='sigmoid')

    weights = classifier.train(X, Y, learning_rate=0.001, epochs=1)
    print(weights)

    # arr1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]).T
    # arr2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).T
    # print((arr1 - arr2))
    # print(np.append(arr1, np.ones(shape=(1, 4)), axis=0))

    # print(arr1 * arr2)
