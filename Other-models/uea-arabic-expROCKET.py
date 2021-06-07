import sys
sys.path.append('./')
from datasets.ucr_uWaveGes import load_data
from rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket
import numpy as np


(X_train, y_train, pic_train), (X_test, y_test, pic_test) = load_data()

X_train=X_train.astype(np.float64)
X_test =X_test.astype(np.float64)

print(X_train.dtype) 
rocket = Rocket()
rocket.fit(X_train)
X_train_transform = rocket.transform(X_train)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(X_train_transform, y_train)

X_test_transform = rocket.transform(X_test)
print(classifier.score(X_test_transform, y_test))


