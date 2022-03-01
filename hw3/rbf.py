import math
import numpy as np
import random

random.seed(2)
np.random.seed(22)


class Rbfn:
    def fit(self, train_x, train_y, strategy='interpolation', regularization=0, std=0, center_num=20):
        if strategy == 'interpolation':
            return self.exact_interpolation(train_x, train_y, regularization, std)
        elif strategy == 'fix':
            return self.fixed_selected_random(train_x, train_y, center_num, std)
        elif strategy == 'k_mean':
            return self.k_mean(train_x, train_y, center_num)

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

    def fixed_selected_random(self, inputs, labels, center_num=20, std=0):
        selected_centers, app_std = self.select_centers(
            inputs, center_num=center_num)

        if std == 0:
            std = app_std
            print(f'The appropriate widths is {std}')

        interpolation_matrix = self.build_interpolation_matrix(
            inputs, selected_centers, std)

        weights = np.dot(np.dot(np.linalg.inv(np.dot(
            interpolation_matrix.T, interpolation_matrix)), interpolation_matrix.T), labels)

        approximator = self.get_approximator(selected_centers, weights, std)

        return approximator

    def k_mean(self, inputs, labels, center_num=2, std=0.1):
        selected_centers = self.k_mean_selection(inputs, center_num)

        interpolation_matrix = self.build_interpolation_matrix(
            inputs, selected_centers, std)

        weights = np.dot(np.dot(np.linalg.inv(np.dot(
            interpolation_matrix.T, interpolation_matrix)), interpolation_matrix.T), labels)

        approximator = self.get_approximator(selected_centers, weights, std)

        return (approximator, np.array(selected_centers))

    def k_mean_selection(self, inputs, center_num=2):
        # inputs: (sample, feature)
        # labels: (sample, 1)
        feature_num = inputs.shape[1]
        selected_centers = np.random.randn(center_num, feature_num)
        center_clusters = [[] for _ in range(0, center_num)]

        iter_num = 0

        while (iter_num < 100):
            for idx in range(0, inputs.shape[0]):
                data = inputs[idx]

                # get the closest centers
                closest_distance = math.inf
                closest_center_idx = 0
                for center_idx in range(0, selected_centers.shape[0]):
                    center = selected_centers[center_idx]
                    distance = np.linalg.norm(center - data)

                    if (distance < closest_distance):
                        # update the closest center
                        closest_distance = distance
                        closest_center_idx = center_idx

                # assign the data to the closest center
                center_clusters[closest_center_idx].append(data)

            prev_centers = selected_centers.copy()
            # update the center
            for center_idx in range(0, len(center_clusters)):
                cluster = center_clusters[center_idx]
                # print(np.array(cluster).shape)
                # no cluster, then randomize the center again
                if len(cluster) == 0:
                    selected_centers[center_idx] = np.random.randn(
                        1, feature_num)
                else:
                    selected_centers[center_idx] = np.mean(
                        cluster, axis=0).reshape((1, feature_num))

                # clear the clusters
                center_clusters[center_idx] = []

            # check if it is converged
            diff = np.abs(prev_centers - selected_centers).sum() / \
                (center_num * feature_num)
            if diff == 0:
                break

            iter_num += 1

        return selected_centers

    def select_centers(self, inputs, center_num=20):
        # inputs: (sample, feature)
        # labels: (sample, 1)
        random_idx = random.sample(
            range(0, inputs.shape[0], 1), center_num)
        selected_centers = []

        for idx in random_idx:
            selected_centers.append(inputs[idx])
        selected_centers = np.array(selected_centers)

        # find the maximum distance among selected data
        max_distance = 0
        for idx in range(0, len(selected_centers)):
            data = selected_centers[idx]
            for idx_ in range(0, len(selected_centers)):
                center = selected_centers[idx_]

                distance = np.linalg.norm(center - data)
                if distance > max_distance:
                    max_distance = distance

        std = max_distance / math.sqrt(2 * center_num)

        return (selected_centers, std)

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
            # hidden_output: (tes_sample, hidden_size)
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

    def get_classification_score_one_hot(self, outputs, labels):
        predictions = np.array([np.argmax(x)
                                for x in outputs]).reshape((labels.shape))
        error = predictions - labels
        return len(np.where(error == 0)[0]) / len(error)

    def get_classification_score(self, tr_outputs, te_outputs, tr_labels, te_labels):
        max_output = np.amax(tr_outputs, axis=0)[0]
        min_output = np.amin(tr_outputs, axis=0)[0]

        train_sample_num = tr_outputs.shape[0]
        test_sample_num = te_outputs.shape[0]

        tr_acc = []
        te_acc = []
        thrs = []

        for idx in range(0, train_sample_num):
            threshold = (max_output - min_output) * idx / 1000 + min_output
            thrs.append(threshold)

            tr_accurracy = (sum(tr_labels[tr_outputs < threshold] == 0) +
                            sum(tr_labels[tr_outputs >= threshold] == 1)) / train_sample_num
            te_accurracy = (sum(te_labels[te_outputs < threshold] == 0) +
                            sum(te_labels[te_outputs >= threshold] == 1)) / test_sample_num

            tr_acc.append(tr_accurracy)
            te_acc.append(te_accurracy)

        return (tr_acc, te_acc, thrs)

    def gaussian(self, x, std=0.1):
        return math.exp(-(x**2 / (2*std**2)))


if __name__ == '__main__':
    print('testing...')
    vec1 = np.array([[1, 2, 3]])
    vec2 = np.array([[3, 5, 4]])
    print(vec1[vec2 < 5], sum(vec1[vec2 < 5] == 1))
    print(np.linalg.norm(vec1 - vec2))
    print(-(1**2 / (2*0.1**2)), 2*(0.1**2))
    print(math.exp(-(1**2 / 2*(0.1**2))))

    g = np.random.normal(loc=0, scale=1, size=10).reshape((10, 1))
    # g = np.reshape(g, (10, 1))
    print(g.shape)
    print(random.sample(range(0, 40, 1), 20))

    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[0.9, 2.1], [3.1, 4.1]])
    print(np.mean(arr1, axis=0).reshape((1, 2)).shape)
    print([[] for i in range(0, 3)])

    print(np.abs(arr1 - arr2).sum())
    print(arr2[:, 0])
