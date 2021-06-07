import sys
sys.path.append('./')
from datasets.ucr_uWaveGes import load_data

from features import *
import pandas as pd
import numpy as np


def column_name(axis_name):
    """
    Return the columns name for the datset
    Parameters:
        axis_name (list): list of axis names

    Returns:
        list: the name of column
    """
    features = ['mean', 'count above mean', 'count below mean', 'mean absolute diff', 'sum absolute diff', 'median',
                'sum', 'absolute energy', 'standard deviation', 'variation coefficient', 'variance', 'skewness',
                'kurtosis', 'number peaks', 'maximum', 'minimum', '25quantile', '75quantile', 'Complexity-Invariant Distance ']
    col_names = []

    for axis in axis_name:
        for feature in features:
            col_names.append(axis+'_'+feature)
    col_names.append('label')
    return col_names

def extract_features(X):
    """
    Return extracted features for X
    Parameters:
        X (3darray): a time-series dat
    Return
        2darray: hand-craft features for X
    """
    if len(X.shape) != 3:
        raise ValueError("X needs to have three deimensions, however, it has " + str(len(X.shape)))
    m , n, k = X.shape

    X_extracted = np.empty((0, 19*k), float)
    print(X_extracted.shape)
    for idx in range(m):
        temp = []
        for axis in range(k):
            temp.append(exteact_all_features(X[idx,:,axis]))
        temp = np.array(temp)
        X_extracted = np.append(X_extracted, temp.reshape(1,np.product(temp.shape)), axis=0)
    return X_extracted


def build_dataset(X, y, axis_names, file_name):
    """
    Computes hand craft features for a given time-series dataset

    Parameters:
        X (ndarray): a time-series data
        y (1darray): an array of label for each sequence
        axis_names: name of each axis
    Returen:
        saved csv file in dataset file
    """
    X_extracted = extract_features(X)
    y = np.expand_dims(y, axis=1)
    data = np.hstack((X_extracted, y))
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(df.head)



(x_train, y_train, pic_train), (x_test, y_test, pic_test) = load_data()

print(x_train.shape)
build_dataset(x_train, y_train, ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8','channel9','channel10','channel11','channel12','channel13'], 'uWave_train')
build_dataset(x_test, y_test, ['channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8','channel9','channel10','channel11','channel12','channel13'], 'uWave_test')