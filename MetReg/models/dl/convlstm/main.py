# author: Lu Li
# mail: lilu35@mail2.sysu.edu.cn
import sys

sys.path.append('../ConvRNN/TrajGRU.py')

import numpy as np
import tensorflow as tf
from models.ConvRNN.TrajGRU import TrajGRU
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Dense


def ConvLSTM():

    mdl = Sequential()
    mdl.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
            padding='same',
            activation='relu',))
    #mdl.add(BatchNormalization())
    mdl.add(Dense(1))
    mdl.build((None, 10, 8, 8, 3))
    
    mdl.summary()
    return mdl

class AttConvLSTM(Model): pass

class trajGRU(Model):

    """implement of trajGRU. Can't used for keras API, only for 
    own-designed training process.
    """

    def __init__(self):
        super().__init__()

        self.trajgru = TrajGRU(num_filter=16, b_h_w=(64, 8, 8))
        self.dense = Dense(1)

    def call(self, inputs):
        history_state, prediction = self.trajgru(inputs)
        prediction = self.dense(prediction)
        return prediction



if __name__ == "__main__":

    trajGRU()



