import tensorflow as tf
import os

import sys
sys.path.append('./')
# from datasets.fordA import load_data
from datasets.har import load_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['AUTOGRAPH_VERBOSITY'] = '10'
tf.autograph.set_verbosity(0)
print("TensorFlow version: {}".format(tf.__version__))
print('Tenserflow CUDA is available: {}'.format(tf.test.is_built_with_cuda()))
print("Eager execution: {}".format(tf.executing_eagerly()))

(x_train, y_train, pic_train), (x_test, y_test, pic_test) = load_data()

input_shape = x_train.shape[1:]
nb_classes = y_train.shape[-1]
mini_batch_size = 32
nb_epochs = 1000

input_layer = tf.keras.layers.Input(input_shape)

conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
conv1 = tf.keras.layers.BatchNormalization()(conv1)
conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
conv2 = tf.keras.layers.BatchNormalization()(conv2)
conv2 = tf.keras.layers.Activation('relu')(conv2)

conv3 = tf.keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
conv3 = tf.keras.layers.BatchNormalization()(conv3)
conv3 = tf.keras.layers.Activation('relu')(conv3)

gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv3)

output_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), 
metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
min_lr=0.0001)


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)

history = model.fit(x_train, y_train, 
                    batch_size=mini_batch_size, 
                    epochs=nb_epochs, 
                    validation_data=(x_test,y_test))

# print(history.history)
