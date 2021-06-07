import sys
sys.path.append('./')
from datasets.ucr_uWaveGes import load_data
from pipnet.pip import PIP
from pipnet.trainingG import fit
from pipnet.trainingG import classification_report


(x_train, y_train, pic_train), (x_test, y_test, pic_test) = load_data()
print(pic_train.shape)
parameters = {'encoder_block': {'filters':(16,16), 'kernel_size':3, 'pool_size':4, 'code_size':64},
              'prototype_block': {'n_prototypes':13, 'n_classes':8}}
model = PIP(parameters=parameters)

fit(model, x_train, y_train, pic_train, x_test, y_test, pic_test, optimization_lr=3e-3, 
                  weight_lr=0.08, 
                  treshold=0.1, 
                  weightUpdate_wait=20,
                  early_stopping=15,
                  batch_size=32, 
                  num_epochs=1000,
                  file_name='ucr-uwave')
classification_report(model, x_test, y_test, class_names=[ '1', '2', '3', '4', '5', '6', '7', '8'],file_name='ucr-uwave')

