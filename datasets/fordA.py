"""
UCR-FordA dataset
"""
import os 
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def __get_pic(y, module_path):
    if y == 0:
        return cv2.imread(module_path+'/datasets/pics/fordA/malfunction.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 1:
        return cv2.imread(module_path+'/datasets/pics/fordA/no_malfunction.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    
def __class_to_pic(Y, module_path):
    pics = []
    for y in Y:
        pics.append(__get_pic(y, module_path))
    return np.expand_dims(np.array(pics),3)

def load_data():
    """
    Load and return the UCR-FordA dataset.

    ==============             ==============
    Training Samples total               3601
    Testing Samples total                1320 
    Number of time steps                  500
    Dimensionality                          1
    Number of targets                       2
    ==============             ==============

    # Returns
        Tuple of Numpy arrays: (x_train, y_train, pic_train), (x_test, y_test, pic_test)
    """
    module_path = os.getcwd()

    train = np.genfromtxt(module_path + '/datasets/data/fordA/FordA_TRAIN.tsv',  delimiter="\t")
    test = np.genfromtxt(module_path + '/datasets/data/fordA/FordA_TEST.tsv',  delimiter="\t")
    x_train = np.expand_dims(train[:,1:], axis=2)
    x_test = np.expand_dims(test[:,1:], axis=2)
    y_train = train[:,0]
    y_test = test[:,0]
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    pic_train = __class_to_pic(y_train, module_path)
    pic_test = __class_to_pic(y_test, module_path)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return (x_train, y_train, pic_train), (x_test, y_test, pic_test)

