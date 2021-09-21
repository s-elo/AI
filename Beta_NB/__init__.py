import math
import time
import matplotlib.pyplot as plt


class BetaNB:
    def __init__(self, train, test, beta_hyper):
        self.train = train
        self.test = test

        # get the prior lamada maximum likelihood from the labels
        # namely the probability of spam emails P(y = 1)
        self.lamada_ML = sum(train.labels == 1) / train.sampleNum

        # cache the each P(X_i = 1 | y = _class) result
        self.prob_f_c_cache = []

        self.cache_prob_f_c(beta_hyper)

    # cache the each P(X_i = 1 | y = _class) result
    def cache_prob_f_c(self, beta_hyper):
        for c in range(2):
            _class = []

            for f in range(57):
                _class.append(self.prob_f_c(f, c, beta_hyper))

            self.prob_f_c_cache.append(_class)

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

    # get the probability of the i-th feature == 1 when given corresponding class
    # namely P(X_i = 1 | y = _class)
    def prob_f_c(self, feature_i, _class, beta_hyper):
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

        return ((feature_i_num + beta_hyper) / (_class_num + beta_hyper * 2))

    def posterior_predict(self, test_sample, _class):
        prob = 0

        # traverse the 57 features
        for f in range(len(test_sample)):
            # sum for each feature according the test sample
            if (test_sample[f] == 1):
                prob = prob + math.log(self.prob_f_c_cache[_class][f])
            elif (test_sample[f] == 0):
                prob = prob + math.log(1 - self.prob_f_c_cache[_class][f])

        return prob

    def getErrorRate(self, data='test'):
        errors = 0

        if (data == 'test'):
            data_set = self.test
        else:
            data_set = self.train

        for s in range(data_set.sampleNum):
            sample = data_set.getOneSample(s, 'binary')
            label = data_set.getOneLabel(s)

            if (self.predict(sample, label) == False):
                errors = errors + 1

        return errors / data_set.sampleNum


def beta_NB_simul(train, test, hyper_range=(0, 100)):
    start = time.time()

    beta_hyper = hyper_range[0]

    train_error_rate = []
    test_error_rate = []

    # store as X axis
    beta = []

    while (beta_hyper <= hyper_range[1]):
        model_nb = BetaNB(train, test, beta_hyper)

        train_error = model_nb.getErrorRate('train')
        test_error = model_nb.getErrorRate('test')

        train_error_rate.append(train_error)
        test_error_rate.append(test_error)

        if (beta_hyper == 1 or beta_hyper == 10 or beta_hyper == 100):
            print('===error rate when α = ' + str(beta_hyper) + '===')
            print('train_error_rate:', train_error)
            print('test_error_rate:', test_error)
            print('\n')

        beta.append(beta_hyper)

        beta_hyper = beta_hyper + 0.5

    end = time.time()

    print('run time:', end - start)

    plt.plot(beta, train_error_rate, 'r')
    plt.plot(beta, test_error_rate, 'b')
    plt.ylabel('Error Rate')
    plt.xlabel('Beta Hyperparameter')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return (train_error_rate, test_error_rate, beta)
