import numpy as np
import math


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

    def gaussian(self, x, std=0.1):
        return math.exp(-(x**2 / (2*std**2)))
