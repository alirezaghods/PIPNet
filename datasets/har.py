"""
UCI-HAR datsets
"""
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
SIGNALS = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
           "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
           "total_acc_x_", "total_acc_y_", "total_acc_z_"]


# taken from https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/README.md
def __load_X(X_signal_paths):
    X_signals = []

    for signal_type_path in X_signal_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

def __get_pic(y, module_path):
    if y == 0:
        return cv2.imread(module_path+'/datasets/pics/har/walking.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 1:
        return cv2.imread(module_path+'/datasets/pics/har/walkingUpstairs.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 2:
        return cv2.imread(module_path+'/datasets/pics/har/walkingDownstairs.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 3:
        return cv2.imread(module_path+'/datasets/pics/har/sitting.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 4:
        return cv2.imread(module_path+'/datasets/pics/har/standing.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 
    elif y == 5: 
        return cv2.imread(module_path+'/datasets/pics/har/laying.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255. 

def __class_to_pic(Y, module_path):
    pics = []
    for y in Y:
        pics.append(__get_pic(y, module_path))
    return np.expand_dims(np.array(pics),3)

def load_data():
    """
    Load and return the UCI-HAR dataset.

    ==============             ==============
    Training Samples total               7352
    Testing Samples total                2947 
    Number of time steps                  128
    Dimensionality                          9
    Number of targets                       6
    ==============             ==============

    # Returns
        Tuple of Numpy arrays: (x_train, y_train, pic_train), (x_test, y_test, pic_test)
    """
    module_path = os.getcwd()
    train_paths = [module_path + '/datasets/data/har/train/Inertial Signals/' + signal + 'train.txt' for signal in SIGNALS]
    test_paths = [module_path + '/datasets/data/har/test/Inertial Signals/' + signal + 'test.txt' for signal in SIGNALS]

    x_train = __load_X(train_paths)
    x_test = __load_X(test_paths)
    y_train = np.loadtxt(module_path+'/datasets/data/har/train/y_train.txt',  dtype=np.int32)
    y_test = np.loadtxt(module_path+'/datasets/data/har/test/y_test.txt', dtype=np.int32)

    y_train = y_train-1
    y_test = y_test-1

    pic_train = __class_to_pic(y_train, module_path)
    pic_test = __class_to_pic(y_test, module_path)

    # y_train = to_categorical(y_train, num_classes=6)
    # y_test = to_categorical(y_test, num_classes=6)

    return (x_train, y_train, pic_train), (x_test, y_test, pic_test)



