from PCA import pca_simul
from LDA import lda_simul
from SVM import svm_simul
import sys

args = sys.argv[1]

if (args == 'pca'):
    pca_simul()
elif (args == 'lda'):
    lda_simul()
elif (args == 'svm'):
    svm_simul()
else:
    print('please enter a right command')
    