import math
import numpy as np


class KNearest:
    def __init__(self, train, test, K):
        self.train = train
        self.test = test

        self.K = K

    def get_K_nearest(self, sample):
        # to record the number of being class1 among all the K_nearest samples
        c1_ret = 0
        # to record the number of being class2
        c2_ret = 0

        # store the distances
        dis = []

        for i in range(self.train.sampleNum):
            model_sample = self.train.getOneSample(i, 'log')
            # get the corresponding label
            label = self.train.getOneLabel(i)

            dis_cur = self.get_distance(sample, model_sample)
            # store the dis and the corresponding label
            dis.append({
                'dis': dis_cur,
                'label': label
            })

        # sort it and slice the first K elements
        K_nearest = sorted(dis, key=lambda x: x['dis'])[0:self.K]

        return K_nearest

    def predict(self, sample):
        K_nearest = self.get_K_nearest(sample)

        # get the number of the class1
        # since the label is 1, so the sum should be the number
        c1_num = sum(list(map(lambda x: x['label'], K_nearest)))
        # the rest should be class2 whose value is 0
        c2_num = self.K - c1_num

        if (c1_num > c2_num):
            return 1
        else:
            return 0

    def getErrorRate(self, data='test'):
        error_num = 0

        if (data == 'test'):
            data_set = self.test
        else:
            data_set = self.train

        for i in range(data_set.sampleNum):
            sample = data_set.getOneSample(i)
            label = data_set.getOneLabel(i)

            ret = self.predict(sample)

            if (ret != label):
                error_num = error_num + 1

        return error_num / data_set.sampleNum

    def get_distance(self, sample1, sample2):
        return np.sqrt(np.sum(np.square(sample1, sample2)))


def K_nearest_simul(train, test, K=None):
    if (K != None):
        K_model = KNearest(train, test, K)
        print(K_model.getErrorRate())

        return

    step = 1
    K = 1

    # error rate for different K
    train_ret = []
    test_ret = []

    # to store the K value as the X axis
    K_label = []

    while (K <= 20):
        K_model = KNearest(train, test, K)

        train_error_rate = K_model.getErrorRate('train')
        test_error_rate = K_model.getErrorRate('test')

        train_ret.append(train_error_rate)
        test_ret.append(test_error_rate)

        K_label.append(K)

        if (K % 5 == 0):
            print('K:', K, 'train_error_rate:', train_error_rate,
                  'test_error_rate:', test_error_rate)
        if (K == 10):
            step = 5

        K = K + step

    return (train_ret, test_ret, K_label)
