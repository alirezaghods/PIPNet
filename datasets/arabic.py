"""
UCR-FordA dataset
"""
import os 
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def __get_pic(y, module_path):
    if y == 10:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_0.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 1:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_1.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 2:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_2.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 3:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_3.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 4:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_4.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 5:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_5.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 6:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_6.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 7:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_7.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 8:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_8.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 9:
        return cv2.imread(module_path+'/datasets/pics/arabic/num_9.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
def __class_to_pic(Y, module_path):
    pics = []
    for y in Y:
        pics.append(__get_pic(y, module_path))
    return np.expand_dims(np.array(pics),3)

def load_data():
    """
    Load and return the UCR-FordA dataset.

    ==============             ==============
    Training Samples total               6600
    Testing Samples total                2200 
    Number of time steps                   93
    Dimensionality                         13
    Number of targets                      10
    ==============             ==============

    # Returns
        Tuple of Numpy arrays: (x_train, y_train, pic_train), (x_test, y_test, pic_test)
    """
    module_path = os.getcwd()

    X_train = np.load(module_path + '/datasets/data/arabic/x_train.npy')
    y_train = np.load(module_path + '/datasets/data/arabic/y_train.npy')
    X_test = np.load(module_path + '/datasets/data/arabic/x_test.npy')
    y_test = np.load(module_path + '/datasets/data/arabic/y_test.npy')
    pic_train = __class_to_pic(y_train, module_path)
    pic_test = __class_to_pic(y_test, module_path)

    y_train = y_train-1
    y_test = y_test-1
    # y_train = to_categorical(y_train, num_classes=10)
    # y_test = to_categorical(y_test, num_classes=10)

    return (X_train, y_train, pic_train), (X_test, y_test, pic_test)

