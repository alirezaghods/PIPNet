from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from pipnet.layers.encoder import EncoderBlock
from pipnet.layers.generator import GeneratorBlock
from pipnet.layers.prototype import PrototypeBlock

class PIP(tf.keras.Model):
    """
    PIP Pictorial Interpratable Prototype Learning Model
    """
    def __init__(self, parameters):
        super(PIP, self).__init__(name='')

        self.encoder_block = EncoderBlock(filters=parameters['encoder_block']['filters'],
                                          kernel_size=parameters['encoder_block']['kernel_size'],
                                          pool_size=parameters['encoder_block']['pool_size'],
                                          code_size=parameters['encoder_block']['code_size'])
        self.generator_block = GeneratorBlock()
        self.prototype_block = PrototypeBlock(n_prototypes=parameters['prototype_block']['n_prototypes'], 
                                              n_features=parameters['encoder_block']['code_size'], 
                                              n_classes=parameters['prototype_block']['n_classes'])

    def encode(self, X):
        return self.encoder_block(X)
    def generate(self, z):
        return self.generator_block(z)
    def classification(self, z):
        return self.prototype_block(z)




