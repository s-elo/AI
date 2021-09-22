import math
import time
import numpy as np
import matplotlib.pyplot as plt


class BetaNB:
    def __init__(self, train, test, hyper_range=(0, 100.5)):
        self.train = train
        self.test = test

        # get the prior lamada maximum likelihood from the labels
        # namely the probability of spam emails P(y = 1)
        self.lamada_ML = sum(train.labels == 1) / train.sampleNum

        # cache the each P(X_i = 1 | y = _class) result
        # (2, 57, 201) 201 different beta_hypers
        self.prob_f_c_cache = []

        # {0, 0.5, 1, 1.5, 2, · · · , 100}
        self.beta_hypers = np.arange(hyper_range[0], hyper_range[1], 0.5)

        self.cache_prob_f_c()

    # cache the each P(X_i = 1 | y = _class) result
    def cache_prob_f_c(self):
        for c in range(2):
            _class = []

            for f in range(57):
                _class.append(self.prob_f_c(f, c))

            self.prob_f_c_cache.append(_class)

    def predict(self, sample, label):
        prior_spam = math.log(self.lamada_ML)
        prior_not_spam = math.log(1 - self.lamada_ML)

        # length == 201 for different beta-hypers
        posterior_predict_spam = self.posterior_predict(sample, 1)
        posterior_predict_not_spam = self.posterior_predict(sample, 0)

        # length == 201 for different beta-hypers
        prob_class_spam = prior_spam + posterior_predict_spam
        prob_class_not_spam = prior_not_spam + posterior_predict_not_spam

        # let the one with higher prob as the result
        # two np lists compare, get a result array
        predictions = (prob_class_spam >= prob_class_not_spam) * 1

        # return the array of if it is an error of not
        # for 201 different beta_hypers. 1 is right, 0 is an error
        return (predictions == label) * 1

    # get the probability of the i-th feature == 1 when given corresponding class
    # namely P(X_i = 1 | y = _class)
    def prob_f_c(self, feature_i, _class):
        # the total number of labels == _class
        _class_num = 0
        # the total number of the i-th feature == 1 when class is _class
        feature_i_num = 0

        for c in range(self.train.sampleNum):
            label = self.train.getOneLabel(c)

            if (label == _class):
                _class_num = _class_num + 1

                # get the corresponding sample of the c-th label
                sample = self.train.getOneSample(c, 'binary')

                if (sample[feature_i] == 1):
                    feature_i_num = feature_i_num + 1
        
        # length == 201 for different beta-hypers
        return ((feature_i_num + self.beta_hypers) / (_class_num + 2 * self.beta_hypers))

    def posterior_predict(self, test_sample, _class):
        prob = 0

        # traverse the 57 features
        for f in range(len(test_sample)):
            # sum for each feature according the test sample
            if (test_sample[f] == 1):
                prob = prob + np.log(self.prob_f_c_cache[_class][f])
            elif (test_sample[f] == 0):
                prob = prob + np.log(1 - self.prob_f_c_cache[_class][f])

        # length == 201 for different beta-hypers
        return prob

    def getErrorRate(self, data='test'):
        corrects = np.zeros(len(self.beta_hypers))

        if (data == 'test'):
            data_set = self.test
        else:
            data_set = self.train

        for s in range(data_set.sampleNum):
            sample = data_set.getOneSample(s, 'binary')
            label = data_set.getOneLabel(s)

            # 1 is correct, 0 is wrong
            corrects = corrects + self.predict(sample, label)
                
        return 1 - (corrects / data_set.sampleNum)


def beta_NB_simul(train, test, hyper_range=(0, 100.5)):
    start = time.time()

    model_nb = BetaNB(train, test, hyper_range)
    
    train_error_rate = model_nb.getErrorRate('train')
    test_error_rate = model_nb.getErrorRate('test')

    print('\n')
    for i in range(3):
        print('===error rate when α = ' + str(10**i) + '===')
        print('train_error_rate:', train_error_rate[10**i * 2])
        print('test_error_rate:', test_error_rate[10**i * 2])
        print('\n')
    
    end = time.time()

    print('run time:', end - start)

    plt.plot(model_nb.beta_hypers, train_error_rate, 'r')
    plt.plot(model_nb.beta_hypers, test_error_rate, 'b')
    plt.ylabel('Error Rate')
    plt.xlabel('Beta Hyperparameter')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return (train_error_rate, test_error_rate)
