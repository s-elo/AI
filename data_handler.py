from scipy.io import loadmat
import numpy as np;

def loadData():
    dataset = loadmat('./dataset/spamData.mat')

    train_data = dataset['Xtrain']
    train_label = dataset['ytrain']
    test_data = dataset['Xtest']
    test_label = dataset['ytest']

    # create instances for train and test
    train = DataHandler(train_data, train_label)
    test = DataHandler(test_data, test_label)

    return (train, test)

class DataHandler:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
        # log_transform
        self.transform_log = np.log(self.data + 0.1)

        # binarization
        self.transform_binary = np.array(self.data, copy=True)
        self.transform_binary[self.data <= 0] = 0
        self.transform_binary[self.data > 0] = 1

    # get the corresponding transformed data 
    # by using specified transform method
    def getData(self, transform = None):
        if (transform == 'log'):
            return self.transform_log
        elif (transform == 'binary'):
            return self.transform_binary
        else:
            return self.data
    
    # get the number of features
    @property
    def featureNum(self):
        return self.data.shape[1]

    # get the number of samples
    @property
    def sampleNum(self):
        return self.data.shape[0]

    # get the corresponding label
    @property
    def labels(self):
        return self.label

    # get the i-th sample with 57 features 
    def getOneSample(self, i, transform = None):
        if (transform == 'log'):
            return self.transform_log[i]
        elif (transform == 'binary'):
            return self.transform_binary[i]
        else:
            return self.data[i]

    # get the i-th label result
    def getOneLabel(self, i):
        return self.label[i][0]
