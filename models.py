import tensorflow as tf
from tensorflow.keras import layers


class OneHidden(tf.keras.Model):
    def __init__(self, hidden_units, num_classes, use_bias=False):
        super(OneHidden, self).__init__()
        self.hidden_layer = Linear(hidden_units, use_bias, name='hidden_layer')
        self.output_layer = Linear(num_classes, use_bias, name='output_layer')

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        x = tf.nn.relu(x)
        x = self.output_layer(x)
        return x


class Linear(layers.Layer):

    def __init__(self, hidden_units, use_bias=False, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.use_bias = use_bias

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='W')

        if self.use_bias:
            self.b = self.add_weight(shape=(self.hidden_units,),
                                     initializer='random_normal',
                                     trainable=True,
                                     name='b')

    def call(self, inputs):
        if self.use_bias:
            return tf.matmul(inputs, self.W) + self.b
        else:
            return tf.matmul(inputs, self.W)

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'hidden_units': self.hidden_units, 'use_bias': self.use_bias})
        return config
