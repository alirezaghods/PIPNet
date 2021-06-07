"""
UCR-UWaveGesture dataset
"""

import os
import numpy as np
import pandas as pd
from scipy.io import arff
import cv2
from tensorflow.keras.utils import to_categorical


def __get_pic(y, module_path):
    if y == 0:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/1.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 1:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/2.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 2:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/3.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 3:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/4.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 4:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/5.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 5:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/6.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 6:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/7.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 7:
        return cv2.imread(module_path+'/datasets/pics/UWaveGesture/8.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 

def __class_to_pic(Y, module_path):
    pics = []
    for y in Y:
        pics.append(__get_pic(y, module_path))
    return np.expand_dims(np.array(pics),3)


def load_data():
    """
    Load and return the UCR-FordA dataset.

    ==============             ==============
    Training Samples total               120
    Testing Samples total                320 
    Number of time steps                 315
    Dimensionality                       3
    Number of targets                    8
    ==============             ==============

    # Returns
        Tuple of Numpy arrays: (x_train, y_train, pic_train), (x_test, y_test, pic_test)
    """
    module_path = os.getcwd()
    print(module_path)
    train_dim1 = pd.DataFrame(arff.loadarff(module_path + '/datasets/data/UWaveGestureLibrary/UWaveGestureLibraryDimension1_TRAIN.arff')[0])
    train_dim2 = pd.DataFrame(arff.loadarff(module_path + '/datasets/data/UWaveGestureLibrary/UWaveGestureLibraryDimension2_TRAIN.arff')[0])
    train_dim3 = pd.DataFrame(arff.loadarff(module_path + '/datasets/data/UWaveGestureLibrary/UWaveGestureLibraryDimension3_TRAIN.arff')[0])
    test_dim1 = pd.DataFrame(arff.loadarff(module_path + '/datasets/data/UWaveGestureLibrary/UWaveGestureLibraryDimension1_TEST.arff')[0])
    test_dim2 = pd.DataFrame(arff.loadarff(module_path + '/datasets/data/UWaveGestureLibrary/UWaveGestureLibraryDimension2_TEST.arff')[0])
    test_dim3 = pd.DataFrame(arff.loadarff(module_path + '/datasets/data/UWaveGestureLibrary/UWaveGestureLibraryDimension3_TEST.arff')[0])
    
    X_train = np.stack([train_dim1[train_dim1.columns[:315]].to_numpy(),train_dim2[train_dim2.columns[:315]].to_numpy(),train_dim3[train_dim3.columns[:315]].to_numpy()],axis=2)
    X_test = np.stack([test_dim1[test_dim1.columns[:315]].to_numpy(),test_dim2[test_dim2.columns[:315]].to_numpy(),test_dim3[test_dim3.columns[:315]].to_numpy()],axis=2)
    y_train = np.array([int(float(y))-1 for y in list(train_dim1.classAttribute)])
    y_test = np.array([int(float(y))-1 for y in list(test_dim1.classAttribute)])

    pic_train = __class_to_pic(y_train, module_path)
    pic_test = __class_to_pic(y_test, module_path)

    # y_train = to_categorical(y_train, num_classes=8)
    # y_test = to_categorical(y_test, num_classes=8)

    return (X_train, y_train, pic_train), (X_test, y_test, pic_test)

