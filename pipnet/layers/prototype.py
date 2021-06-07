import tensorflow as tf
from pipnet.utils import pairwise_dist
from pipnet.utils import normalize
class PrototypeBlock(tf.keras.layers.Layer):
    def __init__(self, n_prototypes, n_features, n_classes):
        """
        Implimentataion of the Prototype layer as defined in the paper

        # Returen:
            output: class distribution
            prototype_distances
            feature_vector_distances
            prototypes: a tensor (n_prototypes, n_features)
            a: prototype distribution, return the 'a' layer output as describe in the paper
        """
        super(PrototypeBlock, self).__init__(name='')

        self.prototype_feature_vectors = tf.Variable(tf.random.uniform(shape=[n_prototypes, n_features],
                                                                       dtype=tf.float32), 
                                                                       name='prototype_feature_vectors')
        self.a = tf.Variable(tf.random.uniform(shape=[1,n_prototypes], 
                                               dtype=tf.float32),
                                               name='prototype_a')

        self.W = tf.Variable(tf.random.uniform(shape=[n_prototypes, n_classes],
                                               dtype=tf.float32),
                                               name='prototype_W')

    def call(self, input_tensor):
        prototype_distances = pairwise_dist(input_tensor, self.prototype_feature_vectors)       # this is R2 in the paper
        feature_vector_distances = pairwise_dist(self.prototype_feature_vectors, input_tensor)  # this is R1 in the paper
        self.a = normalize(prototype_distances)
        logits = tf.matmul(self.a, self.W, name='logits')
        probability_distribution = tf.nn.softmax(logits=logits, name='probability_distribution')
        return probability_distribution, prototype_distances, feature_vector_distances, self.prototype_feature_vectors, self.a
