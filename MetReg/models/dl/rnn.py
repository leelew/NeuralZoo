from tensorflow.keras import models, Model
from tensorflow.keras import layers
from MetReg.base.base_model import BaseModel

import os

os.environ['TP_CPP_MIN_LOG_LEVEL'] = '3'


class BaseRNNRegressor(Model):

    def __init__(self,
                 hidden_layers_sizes=(64,),
                 activation='relu',):
        super().__init__()
        self.regressor = None
        self.hidden_layers_sizes = hidden_layers_sizes

        self.rnn = []
        for i, n_units in enumerate(self.hidden_layers_sizes):
            self.rnn.append(layers.SimpleRNN(units=n_units))

        self.rnn = layers.SimpleRNN(units=64)
        self.dense = layers.Dense(1)

    def call(self, inputs):
        n_features = inputs.shape[-1]
        n_steps = inputs.shape[-2]

        x = self.rnn(inputs)
        return self.dense(x)


class LSTMRegressor(BaseRNNRegressor):

    def __init__(self):
        super().__init__()

        self.lstm = layers.LSTM(units=64)
        self.dense = layers.Dense(1)

    def call(self, inputs):

        x = self.lstm(inputs)
        return self.dense(x)


class rnn:

    def __init__(self): pass

    def __call__(self):

        mdl = tf.keras.models.Sequential()
        mdl.add(tf.keras.layers.RNN(units=64, input_shape=(10, 3)))
        mdl.add(tf.keras.layers.Dense(1))
        mdl.summary()

        return mdl


class gru:

    def __init__(self): pass

    def __call__(self):

        mdl = tf.keras.models.Sequential()
        mdl.add(tf.keras.layers.GRU(units=64, input_shape=(10, 3)))
        mdl.add(tf.keras.layers.Dense(1))
        mdl.summary()

        return mdl


class bilstm:

    def __init__(self): pass

    def __call__(self):

        mdl = tf.keras.models.Sequential()
        mdl.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, input_shape=(10, 3))))
        mdl.add(tf.keras.layers.Dense(1))
        mdl.summary()

        return mdl
