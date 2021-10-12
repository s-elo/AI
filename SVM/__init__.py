import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from loadData import loadData
from PCA import Pca


class Svm:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.pca = Pca(train_data, train_labels, test_data, test_labels)

    def class_simul(self, penalty=1, dim='all'):
        if (dim == 'all'):
            # raw data (N, 1024)
            train_data = self.pca.train_data
            test_data = self.pca.test_data
        else:
            # get the lower dimension data (N, dim)
            train_data, test_data, _ = self.pca.getReducedFaces(dim)

        # labels
        train_labels = self.pca.train_labels
        test_labels = self.pca.test_labels

        # construct a svm model for training data
        classifier = svm.SVC(C=penalty, kernel='linear',
                             decision_function_shape='ovr')

        # fit the trianing data
        classifier.fit(train_data, train_labels)

        # classify the test data
        prediction = classifier.predict(test_data)
        accuracy = accuracy_score(test_labels, prediction)

        return accuracy


def svm_simul():
    random_state = 20

    train_data, train_labels, test_data, test_labels = loadData(
        random_state=random_state)

    svm = Svm(train_data, train_labels, test_data, test_labels)

    C = [0.01, 0.1, 1, 0.00001, 0.000001]
    dim = [80, 200, 'all']

    for c in C:
        print('==============C = ' + str(c) + '==============')
        for d in dim:
            accuracy = svm.class_simul(penalty=c, dim=d)
            print('===Dimension = ' + str(d) + ':===')
            print('Accuracy on the test data: %.6f%%' % (accuracy*100))

        print('\n')
