import math
import numpy as np


class Rbfn:
    def fit(self, train_x, train_y, strategy='interpolation'):
        if strategy == 'interpolation':
            return self.exact_interpolation(train_x, train_y)
        elif strategy == 'fix':
            return self.fixed_selected_random(train_x, train_y)

    def exact_interpolation(self, inputs, labels, std=0.1):
        # inputs: (sample, feature)
        # labels: (sample, 1)
        interpolation_matrix = np.zeros(
            shape=(inputs.shape[0], inputs.shape[0]))

        for idx in range(0, len(inputs)):
            # data: (1, feature)
            data = inputs[idx]

            for idx_ in range(0, len(inputs)):
                center = inputs[idx_]

                distance = np.linalg.norm(data - center)
                interpolation_matrix[idx][idx_] = self.gaussian(distance, std)

        weights = np.dot(np.linalg.inv(interpolation_matrix), labels)

        def approximator(test_inputs):
            # hidden_output: (tes_sample, hidden_size(train_sample))
            hidden_output = np.zeros(
                shape=(test_inputs.shape[0], weights.shape[0]))

            for idx in range(0, len(test_inputs)):
                test_data = test_inputs[idx]

                for idx_ in range(0, len(inputs)):
                    center = inputs[idx_]

                    distance = np.linalg.norm(test_data - center)
                    hidden_output[idx][idx_] = self.gaussian(distance, std)

            # outputs: (test_sample, 1)
            return np.dot(hidden_output, weights)

        return approximator

    def fixed_selected_random(self, inputs, lables):
        pass

    def gaussian(self, x, std=0.1):
        # x: (1, feature)
        return math.exp(-(x**2 / (2*std**2)))


if __name__ == '__main__':
    print('testing...')
    vec1 = np.array([[1, 2, 3]])
    vec2 = np.array([[3, 4, 5]])
    print(np.linalg.norm(vec1 - vec2))
    print(-(1**2 / (2*0.1**2)), 2*(0.1**2))
    print(math.exp(-(1**2 / 2*(0.1**2))))

    g = np.random.normal(loc=0, scale=1, size=10).reshape((10, 1))
    # g = np.reshape(g, (10, 1))
    print(g.shape)
