import numpy as np
import tensorflow as tf

from MetReg.base import BaseLayer, BaseConf

class RnnConf(BaseConf):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default(self):
        pass


class rnn(BaseLayer):

    def __init__(self, layer_conf):
        super().__init__(layer_conf)

        self.rnn = tf.keras.layers.RNN()
    
    def forward(self, inputs):
        return self.rnn(inputs)
