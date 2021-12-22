
import tensorflow as tf
from factory.layers.ConvLSTM import ConvLSTM
from google.protobuf.message import DecodeError
from tensorflow.keras import Model, layers
from tensorflow.python.keras.layers.convolutional import Conv


class Encoder(layers.Layer):
    def __init__(self):

        self.cell_list = [
            ConvLSTM(in_channel=8, filters=16, kernel_size=3, strides=1, padding='same'), 
            ConvLSTM(in_channel=16, filters=64, kernel_size=3, strides=1, padding='same'), 
            ConvLSTM(in_channel=64, filters=64, kernel_size=3,strides=1, padding='same')
            ]
        super().__init__()

    def call(self, inputs):
        hidden_state = []

        for i in range(len(self.cell_list)):
            inputs, state_stage =  self.cell_list[i](inputs)
            print('encoder')
            print(inputs.shape)
            print(state_stage[0].shape)
            hidden_state.append(state_stage)

        return hidden_state
        # [(h1: S*H*W*16, c1), (h2 S*H*W*64, c2), (h3 S*H*W*64, c3)]

class Forecaster(layers.Layer):
    def __init__(self):

        self.cell_list = [
            ConvLSTM(in_channel=64, filters=64, kernel_size=3, strides=1, padding='same'), 
            ConvLSTM(in_channel=64,filters= 64, kernel_size=3, strides=1, padding='same'), 
            ConvLSTM(in_channel=64, filters=16, kernel_size=3, strides=1, padding='same')
            ]
        super().__init__()

    def call(self, hidden_states):
        inputs, states = self.cell_list[0](None, hidden_states[-1])
        print('decoder')
        print(inputs.shape)

        for i in range(len(self.cell_list)):
            inputs, states = self.cell_list[i](inputs, hidden_states[2-i])
            print(inputs.shape)

        return inputs


class EF(layers.Layer):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Forecaster()
        super().__init__()

    def call(self, inputs):
        hidden_states = self.encoder(inputs)
        outputs = self.decoder(hidden_states)

        return outputs

