import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import ConvLSTM2D, SeparableConv2D, Conv2D, Conv3D, Softmax, Input, Dense, Reshape, BatchNormalization
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell
import numpy as np
from tensorflow.python.keras.layers.core import Dense

from tensorflow.keras.optimizers import Adam
from utils.loss import MaskMSELoss, MaskSSIMLoss

class AttentionDecoder(layers.Layer):
    def __init__(self,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):

        self.convlstm = ConvLSTM2DCell(16, 3, padding='same')
        self.conv = Conv2D(16, 3, padding='same')
        self.recurrent_conv = SeparableConv2D(16, 3, padding='same')
        self.recurrent_conv1 = SeparableConv2D(16, 3, padding='same')

        self.conv1 = SeparableConv2D(16, (3, 3), padding='same')
        self.conv2 = Conv3D(16, (1, 3, 3), padding='same')
        self.conv3 = Conv3D(16, (1, 3, 3), padding='same')
        self.conv4 = Conv2D(16, (3, 3), padding='same')

        self.dense = Dense(1)
        self.built = True

    def attention(self, query, value):
        """
        query: hidden state (112, 112, n)
        value: encoder hidden state (t, 112, 112, n)

        adopted from Zhu et al. 2019
        """
        a = self.conv1(query)  # (112,112,16)
        m = a[:,tf.newaxis]
        m = tf.tile(m, [1, value.shape[1], 1, 1, 1])
        b = self.conv2(value)  # (t, 112, 112, 16)
        c = tf.multiply(m, b)  # (t, 112, 112, 16)
        d = self.conv3(c)  # (t, 112, 112, 16)
        e = d / np.sqrt(112 * 112 * 5)
        e = Softmax(axis=1)(e)  # (t, 112, 112, 16)
        g = tf.math.reduce_sum(tf.multiply(value, e),
                               axis=1)  # (112, 112, 16)
        h = self.conv4(g)  # (112, 112, 16)
        j = tf.math.tanh(h)  # (112, 112, 16)
        k = tf.multiply(j, g) + query  #(112, 112,16)
        return k

    def call(self, inputs, state, **kwargs):
        """
        h,c: (maxtime, 112, 112, 16)
        dec_inputs: (outtime, 112, 112, 5)

        Args:
            enc_outputs ([type]): [description]
            dec_inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        h, c0 = state
        h0 = h[:,-1]
        out = []
        for i in range(7):
            _, [h1, c0] = self.convlstm(inputs[:, i], states=[h0, c0])
            h0 = self.attention(h1, h)
            out.append(self.dense(h0))
            
        out = tf.stack(out, axis=1)
        print(out.shape)
        return out


def seq2seq_attn_ConvLSTM(mask, learning_rate):


    enc_inputs = Input(shape=(7, 112, 112, 8), name='1')
    dec_input = Input(shape=(7, 112, 112, 5), name='2')
    h, h0, c0 = ConvLSTM2D(
        16,
        5,
        return_sequences=True,
        return_state=True,
        #activation='relu',
        #activation='linear', adopt 'tanh'.
        padding='same',
        #use_bias=False, # remove bias before BN could increase perform
        kernel_initializer='he_normal')(enc_inputs)
    c1 = AttentionDecoder()(dec_input, [h, c0])
    bn1 = BatchNormalization(axis=-1)(c1)
    bn1 = tf.nn.relu(
        bn1
    )  # BN before ReLU. advised by Jinjing Pan according to ResNet setting.
    out = Dense(1, activation=None)(bn1)

    model = Model(inputs=[enc_inputs, dec_input], outputs=out)

    model.compile(optimizer=Adam(lr=learning_rate), loss=MaskMSELoss(mask))
    model.summary()
    return model


if __name__ == '__main__':
    seq2seq_attn_ConvLSTM(1,1)

