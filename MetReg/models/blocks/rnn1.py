# -----------------------------------------------------------------------------
#                Recurrent Neural Network Model Repository (RNNMR)            #
# -----------------------------------------------------------------------------
# author: Lu Li                                                               #
# mail: lilu35@mail2.sysu.edu.cn                                              #
# -----------------------------------------------------------------------------
# This Repo contains nearly all RNN and state-of-the-art models, which is     #
# implemented using keras. Additinal infos please see guide.pdf               #
# -----------------------------------------------------------------------------


import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, SimpleRNN
from tensorflow.keras.models import Sequential


class rnn():

    def __init__(self):
        pass

    def __call__(self):
        pass


class lstm(rnn):

    def __init__(self):
        pass

    def __call__(self):
        pass


class gru(rnn):

    def __init__(self):
        pass

    def __call__(self):
        pass


class bilstm(lstm):

    def __init__(self):
        pass

    def __call__(self):
        pass


class Recurrent_Neural_Network():

    """implement of RNN, includes:

        1. vanilla RNN
        2. GRU
        3. LSTM
        4. BiLSTM

    All models were simple model with a encoder layers (e.g., LSTM) and a 
    forecasting layer (i.e., Dense).
    """

    def __init__(self):
        pass

    def RNN(self):

        model = Sequential()
        model.add(SimpleRNN(units=64, activation='tanh', return_sequences=True))
        model.add(Dense(1, activation='tanh'))

        return model

    def GRU(self):

        model = Sequential()
        model.add(GRU(units=64, activation='tanh', return_sequences=True))
        model.add(Dense(1, activation='tanh'))

        return model

    def LSTM(self):

        model = Sequential()
        model.add(LSTM(units=64, activation='tanh', return_sequences=True))
        model.add(Dense(1, activation='tanh'))

        return model

    def BiLSTM(self):

        model = Sequential()
        model.add(Bidirectional(LSTM(
            units=64, activation='tanh', return_sequences=True)))
        model.add(Dense(1, activation='tanh'))

        return model


if __name__ == "__main__":

    model = Recurrent_Neural_Network()
    bilstm = model.BiLSTM()
