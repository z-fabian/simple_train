import tensorflow as tf

class OneHidden(tf.keras.Model):
    def __init__(self, hidden_units, num_classes):
        super(OneHidden, self).__init__()
        self.hidden_units = hidden_units
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_units),
                                 initializer='random_normal',
                                 trainable=True)
        self.V = self.add_weight(shape=(self.hidden_units, self.num_classes),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.matmul(inputs, self.W)
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.V)
        return x
