import math
import numpy as np

class GaussianNB:
    def __init__(self, train, test):
        self.train = train
        self.test = test

        # get the prior lamada maximum likelihood from the labels
        # namely the probability of spam emails P(y = 1)
        self.lamada_ML = sum(train.labels == 1) / train.sampleNum

        self.cache_ML_mean_var = self.cache_ML_mean_var()

    # cache the ML estimate of the class conditional mean and variance of each feature
    def cache_ML_mean_var(self):
        cache = []

        for c in range(2):
            _class = []

            for f in range(57):
                _class.append(self.ML_f_c(f, c))

            cache.append(_class)

        return cache

    # get the ML estimate of the i-th feature == 1 when given corresponding class
    # namely the ML estimate mean and variance of P(feature_i = 1 | y = _class)
    def ML_f_c(self, feature_i, _class):
        # store all the i-th feature value given _class
        feature_values = []

        for c in range(self.train.sampleNum):
            if (self.train.getOneLabel(c) == _class):
                # get the corresponding sample of the c-th label
                sample = self.train.getOneSample(c, 'log')

                feature_values.append(sample[feature_i])

        return (np.mean(feature_values), np.var(feature_values))

    def posterior_predict(self, test_sample, _class):
        prob = 0

        # traverse the 57 features
        for f in range(len(test_sample)):
            # get the Ml estimate mean and var of the festure f in _class
            mean, var = self.cache_ML_mean_var[_class][f]

            cur_prob = self.gaussian(test_sample[f], mean, var)
            
            if (cur_prob <= 0):
                return -math.inf
                
            prob = prob + math.log(cur_prob)

        return prob

    def predict(self, sample, label):
        prior_spam = math.log(self.lamada_ML)
        prior_not_spam = math.log(1 - self.lamada_ML)

        posterior_predict_spam = self.posterior_predict(sample, 1)
        posterior_predict_not_spam = self.posterior_predict(sample, 0)

        prob_class_spam = prior_spam + posterior_predict_spam
        prob_class_not_spam = prior_not_spam + posterior_predict_not_spam

        # let the one with higher prob as the result
        if (prob_class_spam >= prob_class_not_spam):
            ret = 1
        else:
            ret = 0

        return (ret == label)

    def getErrorRate(self, data='test'):
        errors = 0

        if (data == 'test'):
            data_set = self.test
        else:
            data_set = self.train

        for s in range(data_set.sampleNum):
            sample = data_set.getOneSample(s)
            label = data_set.getOneLabel(s)

            if (self.predict(sample, label) == False):
                errors = errors + 1

        return errors / data_set.sampleNum

    def gaussian(_, x, mean, variance):
        return (1 / math.sqrt(2 * math.pi * variance)) * math.exp(-0.5 * (x - mean)**2 / variance)
