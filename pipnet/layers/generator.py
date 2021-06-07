import tensorflow as tf

class GeneratorBlock(tf.keras.layers.Layer):
    def __init__(self):
        """
        Implimentataion of the Generator layer as defined in the paper

        # Returen:
            y_pic: tensor of shape(28,28,1), generate a picture for encoding
        """
        super(GeneratorBlock, self).__init__(name='')
        self.dense = tf.keras.layers.Dense(units=7*7*32, activation='relu', name='generator_dense')
        self.reshaping = tf.keras.layers.Reshape(target_shape=(7, 7, 32))
        self.conv2a = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu', name='generator_conv2a')
        self.conv2b = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu', name='generator_conv2b')
        self.conv2c = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME", name='generator_conv2c')
    def call(self, input_tensor):
        x = self.dense(input_tensor)
        x = self.reshaping(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv2c(x)
        return x