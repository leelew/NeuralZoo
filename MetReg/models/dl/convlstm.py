
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Dense
from tensorflow.keras import layers

class BaseConvLSTMRegressor(Model):

    def __init__(self,
                 hidden_layers_sizes=(64,),
                 activation='relu',):
        super().__init__()
        self.regressor = None
        self.hidden_layers_sizes = hidden_layers_sizes

        self.rnn = []
        for i, n_units in enumerate(self.hidden_layers_sizes):
            self.rnn.append(layers.SimpleRNN(units=n_units))

        self.rnn = layers.ConvLSTM2D(units=16,  kernel_size=(3, 3),
            padding='same',
            activation='relu',)
        self.dense = layers.Dense(1)

    def call(self, inputs):
        n_features = inputs.shape[-1]
        n_steps = inputs.shape[-2]

        x = self.rnn(inputs)
        return self.dense(x)


