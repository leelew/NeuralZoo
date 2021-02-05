
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Dense
from tensorflow.keras import layers
from MetReg.models.dl.layers.attention import FeatureAttention, SpatialAttention


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



class AttConvLSTMRegressor():

    def __init__(self): 
        self.feat_att = FeatureAttention()
        self.spatial_att = SpatialAttention()
        self.convlstm = layers.ConvLSTM2D(
            filters=16,  kernel_size=(3, 3),
            padding='same',
            activation='relu',)

    def fit(self, X, y):
        n_timesteps, height, width, n_features = X.shape
        # inputs
        inputs = tf.keras.layers.Input(shape=(n_timesteps, height, width, n_features))

        x = self.feat_att(inputs)
        x = self.spatial_att(x)
        self.




class trajGRURegressor(Model):
    """implement of trajGRU. 
    
    Notes:: Can't used for keras API, only for own-designed training process.
    """

    def __init__(self):
        super().__init__()

        self.trajgru = TrajGRU(num_filter=16, b_h_w=(64, 8, 8))
        self.dense = Dense(1)

    def call(self, inputs):
        history_state, prediction = self.trajgru(inputs)
        prediction = self.dense(prediction)
        return prediction
