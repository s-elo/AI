import math
import numpy as np
import random

random.seed(2)


class Rbfn:
    def fit(self, train_x, train_y, strategy='interpolation', regularization=0, std=0.1):
        if strategy == 'interpolation':
            return self.exact_interpolation(train_x, train_y, regularization, std)
        elif strategy == 'fix':
            return self.fixed_selected_random(train_x, train_y)

    def exact_interpolation(self, inputs, labels, regularization=0, std=0.1):
        # inputs: (sample, feature)
        # labels: (sample, 1)
        interpolation_matrix = self.build_interpolation_matrix(
            inputs, inputs, std)

        if regularization == 0:
            weights = np.dot(np.linalg.inv(interpolation_matrix), labels)
        else:
            weights = np.dot(np.dot(np.linalg.inv(np.dot(
                interpolation_matrix.T, interpolation_matrix) + regularization * np.eye(inputs.shape[0], dtype=int)), interpolation_matrix.T), labels)

        approximator = self.get_approximator(inputs, weights, std)

        return approximator

    def fixed_selected_random(self, inputs, labels):
        center_num = 20
        selected_data, std = self.select_centers(inputs, center_num=center_num)

        interpolation_matrix = self.build_interpolation_matrix(
            inputs, selected_data, std)

        weights = np.dot(np.dot(np.linalg.inv(np.dot(
            interpolation_matrix.T, interpolation_matrix)), interpolation_matrix.T), labels)

        approximator = self.get_approximator(selected_data, weights, std)

        return approximator

    def select_centers(self, inputs, center_num=20):
        # inputs: (sample, feature)
        # labels: (sample, 1)
        random_idx = random.sample(
            range(0, inputs.shape[0], 1), center_num)
        selected_data = []

        for idx in random_idx:
            selected_data.append(inputs[idx])
        selected_data = np.array(selected_data)

        # find the maximum distance among selected data
        max_distance = 0
        for idx in range(0, len(selected_data)):
            data = selected_data[idx]
            for idx_ in range(0, len(selected_data)):
                data_ = selected_data[idx_]

                distance = np.linalg.norm(data_ - data)
                if distance > max_distance:
                    max_distance = distance

        std = max_distance / math.sqrt(2 * center_num)

        return (selected_data, std)

    def build_interpolation_matrix(self, inputs=[], centers=[], std=0.1):
        interpolation_matrix = np.zeros(
            shape=(inputs.shape[0], centers.shape[0]))

        for idx in range(0, len(inputs)):
            # data: (1, feature)
            data = inputs[idx]

            for idx_ in range(0, len(centers)):
                center = centers[idx_]

                distance = np.linalg.norm(data - center)
                interpolation_matrix[idx][idx_] = self.gaussian(distance, std)

        return interpolation_matrix

    def get_approximator(self, centers, weights, std=0.1):
        def approximator(test_inputs):
            # hidden_output: (tes_sample, hidden_size(train_sample))
            hidden_output = np.zeros(
                shape=(test_inputs.shape[0], weights.shape[0]))

            for idx in range(0, len(test_inputs)):
                test_data = test_inputs[idx]

                for idx_ in range(0, len(centers)):
                    center = centers[idx_]

                    distance = np.linalg.norm(test_data - center)
                    hidden_output[idx][idx_] = self.gaussian(distance, std)

            # outputs: (test_sample, 1)
            return np.dot(hidden_output, weights)

        return approximator

    def gaussian(self, x, std=0.1):
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
    print(random.sample(range(0, 40, 1), 20))
