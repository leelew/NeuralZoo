# ==============================================================================
# Keras ConvLSTM2D
#   (the current implementation is derive from torch edition from
#   https://github.com/thuml/predrnn-pytorch/)
#
# Reference:
#   - [Shi et al. 2015]
#
# author: Lu Li
# email: lilu83@mail.sysu.edu.cn
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.python.keras.layers.convolutional_recurrent import \
    ConvLSTM2DCell


class ConvLSTM(layers.Layer): 
    def __init__(self,
                 in_channel: int, 
                 filters: int,
                 kernel_size: int,
                 strides=1,
                 padding='valid',
                 seq_len=7, 
                 batch_size=2):
        super().__init__()

        self.in_channel = int(in_channel)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.padding=padding
        self.strides = strides
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.cell_list = []
        for i in range(seq_len):
            self.cell_list.append(ConvLSTM2DCell(filters=self.filters, 
                                                 kernel_size=self.kernel_size, 
                                                 strides=self.strides,
                                                 padding=self.padding))

    def call(self, inputs=None, states=None):

        if states is None:
            c = tf.zeros((self.batch_size, 112, 112, self.filters))
            h = tf.zeros((self.batch_size, 112, 112, self.filters))
        else:
            h, c = states

        outputs = []
        for i in range(self.seq_len):
            if inputs is None:
                x = tf.zeros((self.batch_size, 112, 112, self.in_channel))
            else:
                x = inputs[:,i]
            
            out, [h, c] = self.cell_list[i](x, [h, c])
            outputs.append(out)

        return tf.stack(outputs, axis=1), [h, c]
