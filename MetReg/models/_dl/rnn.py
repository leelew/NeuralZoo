from numpy.random.mtrand import random_sample
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell
from tensorflow.keras.layers import Dense
import math
import numpy as np
from tensorflow.keras.layers import Conv2D


class ScheduleSamplingSeqSeqConvLSTM(Model):
    """See predrnn.
    
    NOTE: predrnn use weight to control whether use previous true images or previous prediction images. Excellent!
    """
    
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=1, 
                 padding='same', 
                 kernel_initializer='he_normal',
                 reverse_schedule_sampling=True):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.reverse_schedule_sampling = reverse_schedule_sampling

        self.convlstm_cell = ConvLSTM2DCell(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            kernel_initializer=self.kernel_initializer)
        self.dense = Dense(1)#Conv2D(1, 1, 1, 'same', use_bias=False)#Dense(1)

    def call(self, X, mask):
        # X: (s, 14, 112, 112, 1)
        # y: (s, 14, 112, 112, 1)
        # mask: (s, 14, 112, 112, 1)  create by reverse schedule sampling when < 7, and schedule sampling when > 7.

        outputs = []
        h, c = tf.zeros_like(X)[:, 0], tf.zeros_like(X)[:, 0]
        h, c = tf.tile(h, [1, 1, 1, self.filters]), tf.tile(c, [1, 1, 1, self.filters])

        for t in range(14):
        #    # reverse schedule sampling
        #    if self.reverse_schedule_sampling == 1: 
        #        if t == 0:
        #            net = X[:, t]
        #        else:
        #            net = mask[:, t-1]*X[:, t] + (1-mask[:, t-1])*x_gen
        #    else:
        #        if t < 7:
        #            net = X[:, t]
        #        else:
        #            net = mask[:, t-7]*X[:, t] + (1-mask[:, t-7])*x_gen

            _, [h, c] = self.convlstm_cell(X[:, t], states=[h, c])
            x_gen = self.dense(h)
            outputs.append(x_gen)

        return tf.stack(outputs, axis=1)


def reverse_schedule_sampling_exp(itr):
    #TODO: Parameters.
    """see predrnn-pytorch in GitHub."""

    if itr < 25000:
        r_eta, eta = 0.5, 0.5

    elif itr < 50000:
        r_eta = 1.0 - 0.5 * math.exp(-float((itr-25000)/5000))
        eta = 0.5 - (0.5/(25000)*(itr-25000))
    else:
        r_eta, eta = 1.0, 0.0

    r_random_flip = np.random.random_sample((2, 7))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((2, 7))
    true_token = (random_flip < eta)

    ones = np.ones((112, 112, 1))
    zeros = np.zeros((112, 112, 1))

    real_input_flag = []

    for i in range(2):
        for j in range(14):
            if j < 7:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
                    
            else:
                if true_token[i, j - 7]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
    
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (2, 14, 112, 112, 1))
    return real_input_flag

                    

        



                                        
