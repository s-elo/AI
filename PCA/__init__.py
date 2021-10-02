import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

        # get the mean of the training dataset
        self.train_mean = np.mean(train_data, axis=0).reshape(
            1, train_data.shape[1])
        self.test_mean = np.mean(test_data, axis=0).reshape(
            1, test_data.shape[1])

    def dimReduction(self, dataset='train', dim=2):
        # dataset => (N, D)
        if (dataset == 'train'):
            data = self.train_data
            mean = self.train_mean
        else:
            data = self.test_data
            mean = self.test_mean

        sample_num = data.shape[0]

        # get the deviations to the mean (N, D)
        diffs = data - mean

        # compute the SS (D, N)*(N, D)
        S = np.dot(diffs.T, diffs) / sample_num

        # get the eigenValues and eigenVectors
        # eigenVecs[:, i] (column) => eigenVals[i]
        eigenVals, eigenVecs = np.linalg.eig(S)

        # (D, dim)
        select_eigenVecs = eigenVecs[:, 0:dim]

        # (dim, D)*(D, N) = (dim, N).T
        result = np.dot(diffs, select_eigenVecs)

        return (result, select_eigenVecs)

# 40: 27, 80: 23, 200: 23
def pca_simul():
    random_state = 23

    train_data, train_labels, test_data, test_labels = loadData(
        random_state=random_state)

    pca = Pca(train_data, train_labels, test_data, test_labels)
        
    class_simul(pca, dim=2, ispaint=True)
    class_simul(pca, dim=3, ispaint=True)

    accuracy_40 = class_simul(pca, dim=40)
    accuracy_80 = class_simul(pca, dim=80)
    accuracy_200 = class_simul(pca, dim=200)

    print('Dimension 40:')
    print('accuracy on the CMU PIE test images: %.6f%%' %(accuracy_40*100))
    print('===================================')
    print('Dimension 80:')
    print('accuracy on the CMU PIE test images: %.6f%%' %(accuracy_80*100))
    print('===================================')
    print('Dimension 200:')
    print('accuracy on the CMU PIE test images: %.6f%%' %(accuracy_200*100))

    plt.show()


def class_simul(pca, dim=40, ispaint=False):
    # (N, dim), (D, dim)
    train_reduct_imgs,  select_eigenVecs = pca.dimReduction(
        dataset='train', dim=dim)
    test_reduct_imgs = np.dot(
        (pca.test_data - pca.test_mean), select_eigenVecs)

    if (ispaint and dim == 2):
        plt.scatter(
            train_reduct_imgs[:, 0:1], train_reduct_imgs[:, 1:2], c=pca.train_labels, marker=".")
        return
    elif (ispaint and dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_reduct_imgs[:, 0:1], train_reduct_imgs[:, 1:2],
                   train_reduct_imgs[:, 2:3], c=pca.train_labels, marker='.')
        return

    knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn.fit(train_reduct_imgs, pca.train_labels)
    predictions = knn.predict(test_reduct_imgs)
    accuracy = accuracy_score(pca.test_labels, predictions)

    return accuracy
