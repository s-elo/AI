import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from loadData import loadData
from PCA import Pca


class Gmm:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.pca = Pca(train_data, train_labels, test_data, test_labels)

    def class_simul(self, dim='all'):
        if (dim == 'all'):
            # raw data (N, 1024)
            train_data = self.pca.train_data
        else:
            # get the lower dimension data (N, dim)
            train_data, _, _ = self.pca.getReducedFaces(dim)

        gmm = GaussianMixture(n_components=3)
        gmm.fit(train_data)
        ret = gmm.predict(train_data)

        plt.figure()
        plt.scatter(train_data[:, 0], train_data[:, 1], c=ret, marker='.')
        plt.title('dim=' + str(dim))
        
def gmm_simul():
    random_state = 20

    train_data, train_labels, test_data, test_labels = loadData(
        random_state=random_state)

    gmm = Gmm(train_data, train_labels, test_data, test_labels)

    dim = ['all', 200, 80]

    for i in dim:
        gmm.class_simul(i)

    plt.show()
