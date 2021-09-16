from data_handler import *
from Beta_NB import beta_NB_simul
from Gaussian_NB import GaussianNB
from Logistic_reg import LogisticReg, logistic_reg_simul
from K_Nearest import K_nearest_simul
import matplotlib.pyplot as plt

train, test = loadData()

# error_rate_test = beta_NB_simul(train, test, 'test')
# plt.plot(error_rate_test)
# error_rate_train = beta_NB_simul(train, test, 'train')

# plt.plot(error_rate_test, 'r', error_rate_train, 'b')
# plt.ylabel('Error Rate')
# plt.xlabel('Beta Hyperparameter')
# plt.legend(['test', 'train'], loc='upper left')
# plt.show()

# model = GaussianNB(train, test)

# error_rate_test_GNB = model.getErrorRate('test')
# error_rate_train_GNB = model.getErrorRate('train')

# print('test error rate of gaussion NB:', error_rate_test_GNB)
# print('train error rate of gaussion NB:', error_rate_train_GNB)

# error_test_lg, error_train_lg = logistic_reg_simul(train, test, True)

# plt.plot(error_test_lg, 'r', error_train_lg, 'b')
# plt.show()

K_nearest_simul(train, test, 5)
