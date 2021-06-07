import sys
sys.path.append('./')
from datasets.fordA import load_data
from pipnet.pip import PIP
from pipnet.trainingG import fit
from pipnet.trainingG import classification_report


(x_train, y_train, pic_train), (x_test, y_test, pic_test) = load_data()

# for i in range(10):
parameters = {'encoder_block': {'filters':(32,32), 'kernel_size':3, 'pool_size':4, 'code_size':64},
            'prototype_block': {'n_prototypes':2, 'n_classes':2}}
model = PIP(parameters=parameters)

fit(model, x_train, y_train, pic_train, x_test, y_test, pic_test, optimization_lr=3e-3, 
                weight_lr=0.08, 
                treshold=0.1, 
                weightUpdate_wait=20,
                early_stopping=15,
                batch_size=32, 
                num_epochs=1000,
                file_name='ucr-fordA')
classification_report(model, x_test, y_test, class_names=['NoMalfunction', 'Malfunction'],file_name='ucr-fordA')
