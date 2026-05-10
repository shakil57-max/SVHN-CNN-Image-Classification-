import numpy as np
import scipy.io as sio

def load_svhn():
    print("Downloading SVHN dataset...")
    !wget -q http://ufldl.stanford.edu/housenumbers/train_32x32.mat
    !wget -q http://ufldl.stanford.edu/housenumbers/test_32x32.mat

    train_data = sio.loadmat('train_32x32.mat')
    test_data = sio.loadmat('test_32x32.mat')

    x_train = np.transpose(train_data['X'], (3, 0, 1, 2))
    y_train = train_data['y'].flatten()
    x_test = np.transpose(test_data['X'], (3, 0, 1, 2))
    y_test = test_data['y'].flatten()

    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print("Training samples:", x_train.shape[0])
    print("Test samples:", x_test.shape[0])
    
    return x_train, y_train, x_test, y_test
