import numpy as np
import math
import matplotlib.pyplot as plt


class Som:
    def fit(self, train_data, train_label, dim='two', map_size=(10, 10), max_iter=1000):
        if dim == 'two':
            # (map_row_size, map_col_size, feature)
            neuron_weights = self.two_dim_som(train_data, map_size, max_iter)
            classifier = self.get_classifier_two(
                neuron_weights, train_data, train_label)
        elif dim == 'one':
            # (neuron_num, feature)
            neuron_weights = self.one_dim_som(train_data, map_size, max_iter)

        return classifier, neuron_weights

    def get_classifier_two(self, neuron_weights, train_data, train_label):
        row_num, col_num, _ = neuron_weights.shape

        neuron_labels = np.zeros(shape=(row_num, col_num))

        # get the neuron labels
        for row_idx in range(0, row_num):
            for col_idx in range(0, col_num):
                weights = neuron_weights[row_idx, col_idx]

                # compare all the train data and find the most matched one
                closest_distance = math.inf
                closest_idx = 0
                for idx in range(0, train_data.shape[0]):
                    data = train_data[idx]
                    distance = np.linalg.norm(data - weights)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_idx = idx

                neuron_labels[row_idx, col_idx] = train_label[closest_idx]

        def classifier(test_data):
            sample_num = test_data.shape[0]

            outputs = []
            for idx in range(0, sample_num):
                data = test_data[idx]

                # find the winner
                closest_neuron_idx = [0, 0]
                closest_distance = math.inf
                for neuron_row_idx in range(0, row_num):
                    for neuron_col_idx in range(0, col_num):
                        weights = neuron_weights[neuron_row_idx][neuron_col_idx]
                        distance = np.linalg.norm(weights - data)

                        if (distance < closest_distance):
                            closest_neuron_idx = [
                                neuron_row_idx, neuron_col_idx]
                            closest_distance = distance

                outputs.append(
                    neuron_labels[closest_neuron_idx[0], closest_neuron_idx[1]])

            return np.array(outputs).reshape((sample_num, 1))

        return classifier

    def one_dim_som(self, train_data, map_size, max_iter=500):
        np.random.seed(1)
        sample_num, feature_num = train_data.shape

        # initialize neuron weights
        # (neuron_num, feature_num)
        neuron_weights = np.random.randn(map_size[1], feature_num)

        # initialize some hyper params
        init_lr = 0.1
        init_std = math.sqrt((map_size[0]**2) + map_size[1]**2) / 2
        lr_const = sample_num
        std_const = sample_num / math.log(init_std)

        # for later convergence check
        prev_neuron_weights = neuron_weights.copy()

        for step in range(0, max_iter):
            idx = step % sample_num
            data = train_data[idx]

            # get the learning rate and widths for current step
            if step < sample_num:
                # first stage
                lr = init_lr * math.exp(-(idx / lr_const))
                std = init_std * math.exp(-(idx) / std_const)
            else:
                # second stage (convergence stage)
                lr = 0.01
                std = 0.01

            # find the winner
            closest_neuron_idx = 0
            closest_distance = math.inf
            for neuron_idx in range(0, map_size[1]):
                weights = neuron_weights[neuron_idx]
                distance = np.linalg.norm(weights - data)

                if (distance < closest_distance):
                    closest_neuron_idx = neuron_idx
                    closest_distance = distance

            # update the neuron weights
            for neuron_idx in range(0, map_size[1]):
                prev_weights = neuron_weights[neuron_idx]

                # get the effect from the winner
                distance = np.linalg.norm(
                    np.array([neuron_idx]) - np.array([closest_neuron_idx]))
                h = self.gaussian(distance, std=std)

                # update
                neuron_weights[neuron_idx] = prev_weights + \
                    lr * h * (data - prev_weights)

            # check if it reached the convergence every epoch
            if idx == sample_num - 1:
                diff = np.abs(prev_neuron_weights -
                              neuron_weights).sum() / (map_size[0] * map_size[1])
                if (diff < 0.001):
                    break
                else:
                    prev_neuron_weights = neuron_weights.copy()

        print(f'Total iterations: {step + 1}')

        return neuron_weights

    def two_dim_som(self, train_data, map_size, max_iter=500):
        np.random.seed(1)
        sample_num, feature_num = train_data.shape

        # initialize neuron weights
        # (neuron_row_num, neuron_col_num, feature_num)
        neuron_weights = np.random.randn(map_size[0], map_size[1], feature_num)

        # initialize some hyper params
        init_lr = 0.1
        init_std = math.sqrt((map_size[0]**2) + map_size[1]**2) / 2
        lr_const = sample_num
        std_const = sample_num / math.log(init_std)

        # for later convergence check
        prev_neuron_weights = neuron_weights.copy()

        for step in range(0, max_iter):
            idx = step % sample_num
            data = train_data[idx]

            # get the learning rate and widths for current step
            if step < sample_num:
                # first stage
                lr = init_lr * math.exp(-(idx / lr_const))
                std = init_std * math.exp(-(idx) / std_const)
            else:
                # second stage (convergence stage)
                lr = 0.01
                std = 0.01

            # find the winner
            closest_neuron_idx = [0, 0]
            closest_distance = math.inf
            for neuron_row_idx in range(0, map_size[0]):
                for neuron_col_idx in range(0, map_size[1]):
                    weights = neuron_weights[neuron_row_idx][neuron_col_idx]
                    distance = np.linalg.norm(weights - data)

                    if (distance < closest_distance):
                        closest_neuron_idx = [neuron_row_idx, neuron_col_idx]
                        closest_distance = distance

            # update the neuron weights
            for neuron_row_idx in range(0, map_size[0]):
                for neuron_col_idx in range(0, map_size[1]):
                    prev_weights = neuron_weights[neuron_row_idx][neuron_col_idx]

                    # get the effect from the winner
                    distance = np.linalg.norm(
                        np.array([neuron_row_idx, neuron_col_idx]) - np.array([closest_neuron_idx]))
                    h = self.gaussian(distance, std=std)

                    # update
                    neuron_weights[neuron_row_idx][neuron_col_idx] = prev_weights + \
                        lr * h * (data - prev_weights)

            # check if it reached the convergence every epoch
            if idx == sample_num - 1:
                diff = np.abs(prev_neuron_weights -
                              neuron_weights).sum() / (map_size[0] * map_size[1])
                if (diff < 0.001):
                    break
                else:
                    prev_neuron_weights = neuron_weights.copy()

        print(f'Total iterations: {step + 1}')

        return neuron_weights

    def draw_neurons(self, neuron_weights, start_pos=(0, 0), row_num=8, col_num=8, set_label=False):
        cur_row, cur_col = start_pos
        if cur_row + 1 == row_num and cur_col + 1 == col_num:
            return

        weights = neuron_weights[cur_row][cur_col]

        if cur_row + 1 < 8:
            neighbor_weights = neuron_weights[cur_row + 1][cur_col]
            plt_weights = np.array([weights, neighbor_weights])

            if set_label:
                plt.plot(plt_weights[:, 0], plt_weights[:, 1],
                         marker='o', c='blue', label='neuron weights')
            else:
                plt.plot(plt_weights[:, 0], plt_weights[:, 1],
                         marker='o', c='blue')
            self.draw_neurons(neuron_weights, start_pos=(
                cur_row + 1, cur_col), row_num=row_num, col_num=col_num)

        if cur_col + 1 < 8:
            neighbor_weights = neuron_weights[cur_row][cur_col + 1]
            plt_weights = np.array([weights, neighbor_weights])

            plt.plot(plt_weights[:, 0], plt_weights[:, 1],
                     marker='o', c='blue')
            self.draw_neurons(neuron_weights, start_pos=(
                cur_row, cur_col + 1), row_num=row_num, col_num=col_num)

    def draw_weight_map(self, neuron_weights):
        row_num, col_num, _ = neuron_weights.shape
        fig, ax = plt.subplots(10, 10)

        for row_idx in range(0, row_num):
            for col_idx in range(0, col_num):
                weights = neuron_weights[row_idx][col_idx]
                ax[row_idx, col_idx].imshow(
                    weights.reshape((28, 28)), cmap='gray')
                ax[row_idx, col_idx].axis('off')

        fig.suptitle('weights map')

    def gaussian(self, x, std=0.1):
        return math.exp(-(x**2 / (2*std**2)))