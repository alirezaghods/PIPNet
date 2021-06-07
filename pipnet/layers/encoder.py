import tensorflow as tf

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool_size, code_size):
        """
        Implimentataion of the Encoder layer as defined in the paper

        # Arguments:
            filters: list of integers, defining the number of filters
            kernek_size: integer, the kernel size of convolution
            pool_size: integer, the max pool size
            code_size: integer, the length of encoding 
        # Returen:
            z: tensor of shape(), encoding
        """
        super(EncoderBlock, self).__init__(name='')

        F1, F2 = filters
        self.conv1a = tf.keras.layers.Conv1D(F1, kernel_size=kernel_size, padding='same', strides=1, activation='relu', name='encoder_conv1a')
        self.maxpool1a = tf.keras.layers.MaxPooling1D(pool_size, padding='same', name='encoder_maxpol1a')
        self.conv1b = tf.keras.layers.Conv1D(F2, kernel_size=kernel_size, padding='same', strides=1, activation='relu', name='encoder_conv1b')
        self.maxpool1b = tf.keras.layers.MaxPooling1D(pool_size, padding='same',name='encoder_maxpol1b')
        self.flatten = tf.keras.layers.Flatten()
        self.z = tf.keras.layers.Dense(code_size,activation='relu', name='encoding')
    
    def call(self, input_tensor):
        x = self.conv1a(input_tensor)
        x = self.maxpool1a(x)
        x = self.conv1b(x)
        x = self.maxpool1b(x)
        x = self.flatten(x)
        return self.z(x)




          



        
