import numpy as np
import math
import matplotlib.pyplot as plt


class Som:
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

    def gaussian(self, x, std=0.1):
        return math.exp(-(x**2 / (2*std**2)))
