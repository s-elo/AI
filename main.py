from data_handler import *
from Beta_NB import beta_NB_simul
from Gaussian_NB import gaussian_NB_simul
from Logistic_reg import logistic_reg_simul
from K_Nearest import k_nearest_simul
import sys

train, test = loadData()

if (len(sys.argv) == 1):
    print('please give the right command')
    sys.exit()

if (sys.argv[1] == 'beta_NB'):
    beta_NB_simul(train, test)
elif (sys.argv[1] == 'gaussian_NB'):
    gaussian_NB_simul(train, test)
elif (sys.argv[1] == 'logistic_reg'):
    logistic_reg_simul(train, test)
elif (sys.argv[1] == 'k_nearest'):
    k_nearest_simul(train, test)
else:
    print('please give the right command')
