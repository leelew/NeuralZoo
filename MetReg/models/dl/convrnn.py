
import numpy as np
import tensorflow as tf
from MetReg.models.dl.layers.attention import (FeatureAttention,
                                               SpatialAttention)
from MetReg.models.dl.layers.trajgru import TrajGRU
from tensorflow.keras import Model, Sequential, activations, layers


class BaseConvLSTMRegressor(Model):

    def __init__(self):
        super().__init__()
        self.convlstm = layers.ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation='relu', return_sequences=True)
        self.bn1 = layers.BatchNormalization()
        self.convlstm1 = layers.ConvLSTM2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            activation='relu', return_sequences=False)
        self.bn2 = layers.BatchNormalization()
        self.lstm = layers.LSTM(units=8, return_sequences=True)
        self.dense2 = layers.Dense(1)

        self.convlstm2 = layers.ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation='relu', return_sequences=False)
        #self.do = layers.Dropout(rate=0.1)
        self.activation = layers.LeakyReLU(alpha=0.05)

    def call(self, inputs):
        #x = self.convlstm(inputs)
        #x = self.bn1(x)
        #x = self.convlstm1(x)
        #x = self.bn2(x)

        x = self.convlstm2(inputs)
        #x = self.activation(x)
        x = self.bn1(x)
        x = self.dense2(x)

        return x


class AttConvLSTMRegressor(Model):

    def __init__(self):
        super().__init__()
        self.feat_att = FeatureAttention()
        self.spatial_att = SpatialAttention()
        self.convlstm = layers.ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',)
        self.dense = layers.Dense(1)

    def call(self, inputs):
        x = self.feat_att(inputs)
        x = self.spatial_att(x)
        x = self.convlstm(x)
        return self.dense(x)


class trajGRURegressor(Model):
    """implement of trajGRU.

    Notes:: `inputs` must obey tf.dataset structure.
    """

    def __init__(self):
        super().__init__()

        self.trajgru = TrajGRU(
            num_filter=16,
            b_h_w=(64, 8, 8))
        self.dense = layers.Dense(1)

    def call(self, inputs):
        history_state, prediction = self.trajgru(inputs)
        prediction = self.dense(prediction)
        return prediction


class predRNNRegressor(Model):

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        pass
