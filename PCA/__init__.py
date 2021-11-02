import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from loadData import loadData


class Pca:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # (N, D)
        self.train_data = train_data
        self.test_data = test_data
        # (N)
        self.train_labels = train_labels
        self.test_labels = test_labels

        # get the mean of the training dataset (1, D)
        self.train_mean = np.mean(train_data, axis=0).reshape(
            1, train_data.shape[1])
        self.test_mean = np.mean(test_data, axis=0).reshape(
            1, test_data.shape[1])

        # cache the eigenvectors at the beginning
        self.dimReduction()

    def dimReduction(self):
        # dataset => (N, D)
        data = self.train_data
        mean = self.train_mean

        sample_num = data.shape[0]

        # get the deviations to the mean (N, D)
        diffs = data - mean

        # compute the SS (D, N)*(N, D)
        S = np.dot(diffs.T, diffs) / sample_num

        # get the eigenValues and eigenVectors
        # eigenVecs[:, i] (column) => eigenVals[i]
        # eigenVals, eigenVecs = np.linalg.eig(S)

        # sorted from the largest o the smallest
        # idx = eigenVals.argsort()[::-1]
        # eigenVals = eigenVals[idx]
        # eigenVecs = eigenVecs[:,idx]

        # eigenVecs, eigenVals, eigenVecsT
        eigenVecs, _, _ = np.linalg.svd(S, full_matrices=True)

        self.eigenVecs = eigenVecs

        return eigenVecs

    def getReducedFaces(self, dim):
        # (D, D)
        eigenVecs = self.eigenVecs

        # (D, dim)
        select_eigenVecs = eigenVecs[:, 0:dim]

        # (N, D)*(D, dim) = (N, dim)
        train_reduct_imgs = np.dot(
            (self.train_data - self.train_mean), select_eigenVecs)

        test_reduct_imgs = np.dot(
            (self.test_data - self.test_mean), select_eigenVecs)

        return (train_reduct_imgs, test_reduct_imgs, select_eigenVecs)

# 40: 27, 80: 23, 200: 23 (random state)


def pca_simul():
    random_state = 23

    train_data, train_labels, test_data, test_labels = loadData(
        random_state=random_state)

    pca = Pca(train_data, train_labels, test_data, test_labels)

    class_simul(pca, dim=2, ispaint=True)
    class_simul(pca, dim=3, ispaint=True)

    accuracy_PIE_40, accuracy_selfies_40 = class_simul(pca, dim=40)
    accuracy_PIE_80, accuracy_selfies_80 = class_simul(pca, dim=80)
    accuracy_PIE_200, accuracy_selfies_200 = class_simul(pca, dim=200)

    print('Dimension 40:')
    print('accuracy on the CMU PIE test images: %.6f%%' %
          (accuracy_PIE_40*100))
    print('accuracy on my selfies: %.6f%%' % (accuracy_selfies_40*100))
    print('===================================')
    print('Dimension 80:')
    print('accuracy on the CMU PIE test images: %.6f%%' %
          (accuracy_PIE_80*100))
    print('accuracy on my selfies: %.6f%%' % (accuracy_selfies_80*100))
    print('===================================')
    print('Dimension 200:')
    print('accuracy on the CMU PIE test images: %.6f%%' %
          (accuracy_PIE_200*100))
    print('accuracy on my selfies: %.6f%%' % (accuracy_selfies_200*100))

    plt.show()


def class_simul(pca, dim=40, ispaint=False):
    train_reduct_imgs, test_reduct_imgs, select_eigenVecs = pca.getReducedFaces(
        dim)

    if (ispaint and dim == 2):
        plt.scatter(
            train_reduct_imgs[0:500, 0:1], train_reduct_imgs[0:500, 1:2], c=pca.train_labels[0:500], marker=".")

        # selfies
        plt.scatter(
            train_reduct_imgs[-7:, 0:1], train_reduct_imgs[-7:, 1:2], c='red', marker="x")

        return
    elif (ispaint and dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_reduct_imgs[0:500, 0:1], train_reduct_imgs[0:500, 1:2],
                   train_reduct_imgs[0:500, 2:3], c=pca.train_labels[0:500], marker='.')

        # selfies
        ax.scatter(train_reduct_imgs[-7:, 0:1], train_reduct_imgs[-7:, 1:2],
                   train_reduct_imgs[-7:, 2:3], c='red', marker='x')

        # show the eigenfaces
        fig = plt.figure()
        plt.subplot(221)
        plt.imshow(select_eigenVecs[:, 0:1].T.reshape(32, 32), cmap='gray')
        plt.subplot(222)
        plt.imshow(select_eigenVecs[:, 1:2].T.reshape(32, 32), cmap='gray')
        plt.subplot(212)
        plt.imshow(select_eigenVecs[:, 2:3].T.reshape(32, 32), cmap='gray')

        return

    knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn.fit(train_reduct_imgs, pca.train_labels)

    # for the CMU PIE test set
    predictions = knn.predict(test_reduct_imgs[0:-3])
    accuracy_PIE = accuracy_score(pca.test_labels[0:-3], predictions)

    # for my selfies
    predictions = knn.predict(test_reduct_imgs[-3:])
    accuracy_selfies = accuracy_score(pca.test_labels[-3:], predictions)

    return (accuracy_PIE, accuracy_selfies)
