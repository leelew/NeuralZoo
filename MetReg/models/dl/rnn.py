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

        self.lstm = layers.LSTM(units=64, activation='relu')
        self.dense = layers.Dense(1)

    def call(self, inputs):

        x = self.lstm(inputs)
        return self.dense(x)


class GRURegressor(BaseRNNRegressor):

    def __init__(self):
        super().__init__()

        self.gru = layers.GRU(units=64)
        self.dense = layers.Dense(1)

    def call(self, inputs):

        x = self.gru(inputs)
        return self.dense(x)


class BiLSTMRegressor(BaseRNNRegressor):

    def __init__(self):
        super().__init__()

        self.bilstm = layers.Bidirectional(
            layers.LSTM(units=64))

        self.dense = layers.Dense(1)

    def call(self, inputs):

        x = self.bilstm(inputs)
        return self.dense(x)
