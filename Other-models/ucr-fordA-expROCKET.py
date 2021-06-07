import sys
sys.path.append('./')
from datasets.fordA import load_data
from rocket_functions import generate_kernels, apply_kernels
from sklearn.linear_model import RidgeClassifierCV


(x_train, y_train, pic_train), (x_test, y_test, pic_test) = load_data()

(x_train, y_train, pic_train), (x_test, y_test, pic_test) = load_data()
print(x_train.shape)
kernels = generate_kernels(x_train.shape[-1], 10_000)

X_training_transform = apply_kernels(x_train, kernels)
classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, y_train)

X_test_transform = apply_kernels(X_test, kernels)
predictions = classifier.predict(X_test_transform)
