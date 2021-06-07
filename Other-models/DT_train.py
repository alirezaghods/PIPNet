import pandas as pd
import numpy as np

test = pd.read_csv('uWave_test')
train = pd.read_csv('uWave_train')

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from IPython.display import Image 
import pydotplus
import matplotlib.pyplot as plt


clf = DecisionTreeClassifier(max_depth=4,random_state=0)
model = clf.fit(train[train.columns[:-1]], train[train.columns[-1]])
pred = clf.predict(test[test.columns[:-1]])

print(accuracy_score(test[test.columns[-1]], pred))

