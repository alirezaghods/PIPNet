import sys
sys.path.append('./')
from datasets.arabic import load_data
from pipnet.pip import PIP
from pipnet.trainingR2 import fit
from pipnet.trainingR2 import classification_report


(x_train, y_train, pic_train), (x_test, y_test, pic_test) = load_data()

parameters = {'encoder_block': {'filters':(16,16), 'kernel_size':3, 'pool_size':4, 'code_size':64},
              'prototype_block': {'n_prototypes':10, 'n_classes':10}}
model = PIP(parameters=parameters)

fit(model, x_train, y_train, pic_train, x_test, y_test, pic_test, optimization_lr=3e-3, 
                  weight_lr=0.08, 
                  treshold=0.1, 
                  weightUpdate_wait=20,
                  early_stopping=15,
                  batch_size=32, 
                  num_epochs=1000,
                  file_name='uea-arabic')
classification_report(model, x_test, y_test, class_names=[ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],file_name='uea-arabic')

