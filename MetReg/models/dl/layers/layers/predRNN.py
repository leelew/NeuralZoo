# ==============================================================================
# Keras predRNN
#   (the current implementation is derive from torch edition from
#   https://github.com//thuml/predrnn-pytorch/)
#
# Reference:
#   - [Wang et al., 2017] Predrnn: Recurrent neural networks for predictive
#   learning using spatiotemporal lstms.
#
# author: Lu Li
# email: lilu83@mail.sysu.edu.cn
# ==============================================================================

from traceback import print_tb

import tensorflow as tf
from tensorflow import sigmoid, split, tanh
from tensorflow.keras import Input, Model, Sequential, layers
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D,
                                     Dense, ReLU)
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops.gen_math_ops import Mod


class STLSTM2DCell(layers.Layer):
    """Spatial-temporal LSTM

    Args:
      units ([type]): [description]
      kernel_size ([type]): [description]
      stride ([type]): [description]
      layer_norm (bool, optional): [description]. Defaults to True.
      trainable (bool, optional): [description]. Defaults to True.
      name ([type], optional): [description]. Defaults to None.
      dtype ([type], optional): [description]. Defaults to None.
      dynamic (bool, optional): [description]. Defaults to False.
    """
    def __init__(self,
                 units,
                 kernel_size,
                 stride=1,
                 layer_norm=True,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.units = units
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = 'same'  #kernel_size // 2
        self._forget_bias = 1.0
        self.layer_norm = layer_norm

        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):

        if self.layer_norm:
            self.conv_x = Sequential([
                Conv2D(self.units * 7,
                       kernel_size=self.kernel_size,
                       strides=self.stride,
                       padding=self.padding,
                       use_bias=False,
                       activation='linear',
                       input_shape=((112, 112, input_shape[-1]))),
                BatchNormalization(),
                ReLU()
            ])

            self.conv_h = Sequential([
                Conv2D(self.units * 4,
                       kernel_size=self.kernel_size,
                       strides=self.stride,
                       padding=self.padding,
                       use_bias=False,
                       activation='linear'),
                BatchNormalization(),
                ReLU()
            ])

            self.conv_m = Sequential([
                Conv2D(self.units * 3,
                       kernel_size=self.kernel_size,
                       strides=self.stride,
                       padding=self.padding,
                       use_bias=False,
                       activation='linear'),
                BatchNormalization(),
                ReLU()
            ])

            self.conv_o = Sequential([
                Conv2D(self.units,
                       kernel_size=self.kernel_size,
                       strides=self.stride,
                       padding=self.padding,
                       use_bias=False,
                       activation='linear'),
                BatchNormalization(),
                ReLU()
            ])

            self.conv_last = Sequential([
                Conv2D(self.units,
                       kernel_size=1,
                       strides=1,
                       padding=self.padding,
                       use_bias=False,
                       activation='linear'),
                BatchNormalization(),
                ReLU()
            ])
        else:

            self.conv_x = Conv2D(self.units * 7,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride,
                                 padding=self.padding,
                                 use_bias=False)

            self.conv_h = Conv2D(self.units * 4,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride,
                                 padding=self.padding,
                                 use_bias=False)

            self.conv_m = Conv2D(self.units * 3,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride,
                                 padding=self.padding,
                                 use_bias=False)

            self.conv_o = Conv2D(self.units,
                                 kernel_size=self.kernel_size,
                                 strides=self.stride,
                                 padding=self.padding,
                                 use_bias=False)

            self.conv_last = Conv2D(self.units,
                                    kernel_size=1,
                                    strides=1,
                                    padding=self.padding,
                                    use_bias=False)
        self.built = True

    def call(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = tf.split(
            x_concat, 7, axis=-1)

        i_h, f_h, g_h, o_h = tf.split(h_concat, 4, axis=-1)
        i_m, f_m, g_m = tf.split(m_concat, 3, axis=-1)

        i_t = tf.sigmoid(i_x + i_h)
        f_t = tf.sigmoid(f_x + f_h + self._forget_bias)
        g_t = tf.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = tf.sigmoid(i_x_prime + i_m)
        f_t_prime = tf.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = tf.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = tf.concat((c_new, m_new), -1)
        o_t = tf.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * tf.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class predRNN(layers.Layer):
    def __init__(self,
                 num_layers: int,
                 num_hidden: list,
                 kernel_size: int,
                 layer_norm: bool = True,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        #NOTE: num_hidden must be the same for each layers.maybe should let memory could change.
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.layer_norm = layer_norm
        assert len(num_hidden) == num_layers

        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):
        self.b, self.t, self.h, self.w, _ = input_shape
        self.cell_list = []
        self.init_list = []
        for i in range(self.num_layers):
            self.cell_list.append(
                STLSTM2DCell(self.num_hidden[i],
                             kernel_size=self.kernel_size,
                             layer_norm=self.layer_norm))

            self.init_list.append(Dense(self.num_hidden[i]))
        self.dense_last = Dense(1)  #conv_last = Conv2D(1, 1)

        self.built = True

    def call(self, inputs, states=None):
        # states: [h, c, m]  shape as (112, 112, n)

        # init parameters
        h_t, c_t = [], []
        zero = tf.zeros_like(inputs[:, -1])
        for i in range(self.num_layers):
            zeros = self.init_list[i](zero)
            h_t.append(zeros)
            c_t.append(zeros)

        #FIXME: Change to the same manner of h, c
        memory = self.init_list[0](zero)

        if states is not None:
            h_t[0] = states[0]
            c_t[0] = states[1]
            memory = states[2]

        out = []
        # forward for each timestep
        for t in range(self.t):
            frame = inputs[:, t]

            # forward for first layer
            h_t[0], c_t[0], memory = self.cell_list[0](frame, h_t[0], c_t[0],
                                                       memory)

            # forward for rest layers
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i],
                                                           c_t[i], memory)

            # generate output for each timestep
            out.append(self.dense_last(h_t[self.num_layers - 1]))

        return tf.stack(out, axis=1), [h_t[-1], c_t[-1], memory]


if __name__ == '__main__':

    import numpy as np
    inputs = Input((7, 112, 112, 8))
    outputs, states = predRNN(3, [64, 64, 64], kernel_size=5)(inputs)
    outputs, states = predRNN(3, [64, 64, 64], kernel_size=5)(inputs, states)
    mdl = Model(inputs, outputs)
    mdl.summary()
