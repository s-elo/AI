import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from loadData import loadData


class Lda:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # (N, D)
        self.train_data = train_data
        self.test_data = test_data
        # (N)
        self.train_labels = train_labels
        self.test_labels = test_labels

        # except the selfies
        self.total_train_num = len(train_labels) - 7
        self.total_class_num = 25

        # the number of each class is the same except the selfies
        self.Nc = int((self.total_train_num) / (self.total_class_num))

        # (1, D)
        self.total_mean = np.mean(train_data, axis=0).reshape(
            1, train_data.shape[1])
        self.test_mean = np.mean(test_data, axis=0).reshape(
            1, test_data.shape[1])

        # get the means of each class in training set including selfies (class_num, D)
        self.class_means = []

        for i in range(1, 26):
            class_set = train_data[(i-1)*self.Nc: i*self.Nc, :]

            # (1, D)
            mean = np.mean(class_set, axis=0)

            self.class_means.append(mean)

        # (class_num, D) add the selfies mean
        self.class_means.append(np.mean(train_data[-7:, :], axis=0))
        self.class_means = np.array(self.class_means)

        self.getSb()
        self.getSw()

        Sw_Sb = np.dot(np.linalg.inv(self.Sw), self.Sb)
        eigenVecs, _, _ = np.linalg.svd(Sw_Sb, full_matrices=True)

        # cache the eigenVectors at the beginning
        self.eigenVecs = eigenVecs

    def getSb(self):
        # (class_num, D)
        mean_diffs = self.class_means - self.total_mean

        # Nc*(D, class_num)*(class_num, D) = Nc*(D, D) exclude selfies
        self.Sb = np.dot(mean_diffs[0:-1, :].T, mean_diffs[0:-1, :])*self.Nc

        # selfies (D, 1)*(1, D) = (D, D)
        self.Sb = self.Sb + np.dot(mean_diffs[-1:, :].T, mean_diffs[-1:, :])*7

    def getSw(self):
        feature_num = self.train_data.shape[1]
        Sw = np.zeros(shape=(feature_num, feature_num))

        # exclude selfies
        for i in range(1, 26):
            # (25, D)
            data_set = self.train_data[(i-1)*self.Nc: i*self.Nc, :]
            # (1, D)
            mean = self.class_means[i]
            # (25, D)
            diffs = data_set - mean

            Sw = Sw + np.dot(diffs.T, diffs)

        # selfies
        diffs = self.train_data[-7:, :] - self.class_means[-1]
        self.Sw = Sw + np.dot(diffs.T, diffs)

    def getReducedFishes(self, dim):
        # (D, D)
        eigenVecs = self.eigenVecs

        # (D, dim)
        select_eigenVecs = eigenVecs[:, 0:dim]

        # (dim, D)*(D, N) = (N, dim)
        train_reduct_imgs = np.dot(self.train_data - self.total_mean, select_eigenVecs)

        # (N, D)*(D, dim) = (N, dim)
        test_reduct_imgs = np.dot(self.test_data - self.test_mean, select_eigenVecs)

        return (train_reduct_imgs, test_reduct_imgs, select_eigenVecs)

    def class_simul(self, dim, ispaint=False):
        train_reduct_imgs, test_reduct_imgs, _ = self.getReducedFishes(dim)

        if (ispaint and dim == 2):
            plt.scatter(
                train_reduct_imgs[0:-7, 0:1], train_reduct_imgs[0:-7, 1:2], c=self.train_labels[0:-7], marker=".")

            # selfies
            plt.scatter(
                train_reduct_imgs[-7:, 0:1], train_reduct_imgs[-7:, 1:2], c='red', marker="x")

        elif (ispaint and dim == 3):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(train_reduct_imgs[0:-7, 0:1], train_reduct_imgs[0:-7, 1:2],
                       train_reduct_imgs[0:-7, 2:3], c=self.train_labels[0:-7], marker='.')

            # selfies
            ax.scatter(train_reduct_imgs[-7:, 0:1], train_reduct_imgs[-7:, 1:2],
                       train_reduct_imgs[-7:, 2:3], c='red', marker='x')

        knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
        knn.fit(train_reduct_imgs, self.train_labels)

        # for the CMU PIE test set
        predictions = knn.predict(test_reduct_imgs[0:-3])
        accuracy_PIE = accuracy_score(self.test_labels[0:-3], predictions)

        # for my selfies
        predictions = knn.predict(test_reduct_imgs[-3:])
        accuracy_selfies = accuracy_score(self.test_labels[-3:], predictions)

        return (accuracy_PIE, accuracy_selfies)


def lda_simul():
    random_state = 16

    train_data, train_labels, test_data, test_labels = loadData(
        random_state=random_state)

    lda = Lda(train_data, train_labels, test_data, test_labels)

    accuracy_PIE_2, accuracy_selfies_2 = lda.class_simul(dim=2, ispaint=True)
    accuracy_PIE_3, accuracy_selfies_3 = lda.class_simul(dim=3, ispaint=True)
    accuracy_PIE_9, accuracy_selfies_9 = lda.class_simul(dim=9)

    print('Dimension 2:')
    print('accuracy on the CMU PIE test images: %.6f%%' %
          (accuracy_PIE_2*100))
    print('accuracy on my selfies: %.6f%%' % (accuracy_selfies_2*100))
    print('===================================')
    print('Dimension 3:')
    print('accuracy on the CMU PIE test images: %.6f%%' %
          (accuracy_PIE_3*100))
    print('accuracy on my selfies: %.6f%%' % (accuracy_selfies_3*100))
    print('===================================')
    print('Dimension 9:')
    print('accuracy on the CMU PIE test images: %.6f%%' %
          (accuracy_PIE_9*100))
    print('accuracy on my selfies: %.6f%%' % (accuracy_selfies_9*100))

    plt.show()
